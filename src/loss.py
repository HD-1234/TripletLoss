from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

from src.triplet_miner import TripletMiner


class SelectivelyContrastiveTripletLoss(Module):
    def __init__(
            self,
            positive_mining_strategy: str,
            negative_mining_strategy: str,
            temperature: float = 0.1,
            margin: float = 1.0,
            lam: float = 1.0
    ) -> None:
        """
        Initializes the Selectively Contrastive Triplet Loss module.

        Source:
            https://arxiv.org/pdf/2007.12749

        Args:
            positive_mining_strategy (str): Strategy for mining positive samples ('easy' or 'random').
            negative_mining_strategy (str): Strategy for mining negative samples ('hard', 'semi-hard' or 'random').
            temperature (float): Temperature parameter for the softmax function.
            margin (float): Margin parameter for semi-hard negative mining.
            lam (float): Weight for the case that the positives are closer to the anchors than the negatives.
        """
        super(SelectivelyContrastiveTripletLoss, self).__init__()
        self.temperature = temperature
        self.lam = lam
        self.triplet_miner = TripletMiner(
            positive_mining_strategy=positive_mining_strategy,
            negative_mining_strategy=negative_mining_strategy,
            margin=margin
        )

    @staticmethod
    def _calculate_similarity_scores(
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the similarity scores between anchor and positives and anchor and negatives.

        Args:
            anchor (torch.Tensor): Tensor of anchor embeddings.
            positive (torch.Tensor): Tensor of positive embeddings.
            negative (torch.Tensor): Tensor of negative embeddings.

        Returns:
            Tuple: Matrix of similarity scores.
        """
        # Normalize the embeddings for anchors, positives, and negatives
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Calculate the similarity scores between anchors and positives
        positive_sim_scores = torch.matmul(anchor, torch.t(positive))
        positive_scores = positive_sim_scores.diag()

        # Calculate the similarity scores between anchors and negatives
        negative_sim_scores = torch.matmul(anchor, torch.t(negative))
        negative_scores = negative_sim_scores.diag()

        return positive_scores, negative_scores

    @staticmethod
    def _compute_hard_negative_loss(
            positive_scores: torch.Tensor,
            negative_scores: torch.Tensor,
            valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for hard negatives.

        Args:
            positive_scores (torch.Tensor): Scores of the mined positive samples.
            negative_scores (torch.Tensor): Scores of the mined negative samples.
            valid_mask (torch.Tensor): Mask indicating valid samples.

        Returns:
            Tuple: Loss and the number of valid scores.
        """
        # Create a mask for valid hard negatives
        mask = ((negative_scores > positive_scores) | (negative_scores > 0.8)) & valid_mask

        # Compute the loss by summing the valid negative scores
        loss = negative_scores[mask].sum()

        # Count the number of valid scores
        num_valid_scores = mask.int().sum()

        # Handle NaN values or no valid scores by setting the loss to 0
        if torch.isnan(loss) or num_valid_scores == 0:
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        return loss, num_valid_scores

    def _compute_loss_others(
            self,
            positive_scores: torch.Tensor,
            negative_scores: torch.Tensor,
            valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for other non-hard cases.

        Args:
            positive_scores (torch.Tensor): Scores of the mined positive samples.
            negative_scores (torch.Tensor): Scores of the mined negative samples.
            valid_mask (torch.Tensor): Mask indicating valid samples.

        Returns:
            Tuple: Loss and the number of valid scores.
        """
        # Create a mask for valid samples
        mask = (negative_scores <= positive_scores) & (negative_scores <= 0.8) & valid_mask

        # Apply the mask to the positive and negative scores
        positive_scores = positive_scores[mask]
        negative_scores = negative_scores[mask]

        # Apply the temperature scaling to the positive and negative scores
        positive_scores /= self.temperature
        negative_scores /= self.temperature

        # Compute the softmax probabilities for the positive and negative scores
        softmax = torch.exp(positive_scores) / (torch.exp(positive_scores) + torch.exp(negative_scores))

        # Compute the loss by taking the negative log of the softmax probabilities
        loss = -torch.log(softmax).sum()

        # Count the number of valid scores
        num_valid_scores = mask.int().sum()

        # Handle NaN values or no valid scores by setting the loss to 0
        if torch.isnan(loss) or num_valid_scores == 0:
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        return loss, num_valid_scores

    def forward(
            self,
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor,
            anchor_label: torch.Tensor,
            positive_label: torch.Tensor,
            negative_label: torch.Tensor,
            eval_mode: bool = False
    ) -> torch.Tensor:
        """
        Computes the Selectively Contrastive Triplet loss, which is a contrastive loss that penalizes the
        anchor-negative similarity.

        Args:
            anchor (torch.Tensor): Anchor embeddings.
            positive (torch.Tensor): Positive embeddings.
            negative (torch.Tensor): Negative embeddings.
            anchor_label (torch.Tensor): Labels for the anchor embeddings.
            positive_label (torch.Tensor): Labels for the positive embeddings.
            negative_label (torch.Tensor): Labels for the negative embeddings.
            eval_mode (bool): Whether to mine for triplets or not.

        Returns:
            torch.Tensor: The computed loss.
        """
        if not eval_mode:
            # Mine samples and calculate the positive and negative scores
            positive_scores, negative_scores, _, _, valid_pos, valid_neg, _ = self.triplet_miner.mine_triplets(
                anchor=anchor,
                positive=positive,
                negative=negative,
                anchor_label=anchor_label,
                positive_label=positive_label,
                negative_label=negative_label
            )

            # Create a mask to filter out invalid scores
            valid_mask = valid_pos & valid_neg
        else:
            # Calculate the positive and negative scores
            positive_scores, negative_scores = self._calculate_similarity_scores(
                anchor=anchor,
                positive=positive,
                negative=negative,
            )

            # Create a mask to filter out invalid scores (all valid in eval mode)
            valid_mask = torch.ones_like(positive_scores, device=positive_scores.device, dtype=torch.bool)

        # Compute the loss for hard negatives
        loss_hard_negative, hard_negative_valid_scores = self._compute_hard_negative_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            valid_mask=valid_mask
        )

        # Compute the loss for other cases
        loss_other, other_valid_scores = self._compute_loss_others(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            valid_mask=valid_mask
        )

        # Compute the total count of valid cases
        total_valid_scores = hard_negative_valid_scores + other_valid_scores

        # Avoid division by zero
        if total_valid_scores == 0:
            total_valid_scores = 1

        # Compute the final loss
        loss = (loss_other + self.lam * loss_hard_negative) / total_valid_scores

        return loss


class TripletLossWithMargin(Module):
    def __init__(
            self,
            positive_mining_strategy: str,
            negative_mining_strategy: str,
            margin: float = 1.0
    ) -> None:
        """
        Initializes the triplet margin loss.

        Args:
            positive_mining_strategy (str): Strategy for mining positive samples ('easy' or 'random').
            negative_mining_strategy (str): Strategy for mining negative samples ('hard', 'semi-hard', or 'random').
            margin (float): Margin parameter for semi-hard negative mining.
        """
        super(TripletLossWithMargin, self).__init__()
        self.triplet_miner = TripletMiner(
            positive_mining_strategy=positive_mining_strategy,
            negative_mining_strategy=negative_mining_strategy,
            margin=margin
        )
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)

    def forward(
            self,
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor,
            anchor_label: torch.Tensor,
            positive_label: torch.Tensor,
            negative_label: torch.Tensor,
            eval_mode: bool = False
    ) -> torch.Tensor:
        """
        Computes the triplet margin loss for the input tensors.

        Args:
            anchor (torch.Tensor): Anchor embeddings.
            positive (torch.Tensor): Positive embeddings.
            negative (torch.Tensor): Negative embeddings.
            anchor_label (torch.Tensor): Labels for the anchor embeddings.
            positive_label (torch.Tensor): Labels for the positive embeddings.
            negative_label (torch.Tensor): Labels for the negative embeddings.
            eval_mode (bool): Whether to mine for triplets or not.

        Returns:
            torch.Tensor: The computed loss.
        """
        if not eval_mode:
            # Mine the positive and negative indices, get the corresponding masks and the unique indices
            _, _, positive_ind, negative_ind, valid_pos, valid_neg, unique_ind = self.triplet_miner.mine_triplets(
                anchor=anchor,
                positive=positive,
                negative=negative,
                anchor_label=anchor_label,
                positive_label=positive_label,
                negative_label=negative_label
            )

            # Concatenate the embeddings
            embeddings = torch.cat((anchor, positive, negative), 0)

            # Create a mask to filter out invalid indices
            valid_mask = valid_pos & valid_neg

            # Get anchor, positive and negative
            anchor = embeddings[unique_ind]
            positive = embeddings[positive_ind]
            negative = embeddings[negative_ind]

            # Apply mask to anchor, positive and negative
            anchor = anchor[valid_mask]
            positive = positive[valid_mask]
            negative = negative[valid_mask]

        # Compute the loss
        loss = self.loss_fn(anchor, positive, negative)

        # Handle NaN values by setting the loss to 0
        loss = torch.nan_to_num(loss, nan=0.0) if torch.isnan(loss) else loss

        return loss
