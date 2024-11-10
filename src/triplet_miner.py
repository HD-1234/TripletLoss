from typing import Tuple

import torch
import torch.nn.functional as F


class TripletMiner:
    def __init__(self, positive_mining_strategy: str, negative_mining_strategy: str, margin: float = 1.0):
        """
        Initializes the TripletMiner with specified mining strategies and margin.

        Args:
            positive_mining_strategy (str): Strategy for mining positive samples ('easy' or 'random').
            negative_mining_strategy (str): Strategy for mining negative samples ('hard', 'semi-hard' or 'random').
            margin (float): Margin for the triplet loss.
        """
        super(TripletMiner, self).__init__()
        self.positive_mining_strategy = positive_mining_strategy
        self.negative_mining_strategy = negative_mining_strategy
        self.margin = margin

    @staticmethod
    def _filter_duplicates(
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Removes duplicate embeddings and the corresponding labels.

        Args:
            embeddings (torch.Tensor): Tensor of embeddings.
            labels (torch.Tensor): Tensor of labels.

        Returns:
            Tuple: Unique embeddings, their corresponding labels and the unique indices.
        """
        # Find unique embeddings and return the corresponding indices and counts
        _, indices, counts = torch.unique(embeddings, dim=0, sorted=True, return_inverse=True, return_counts=True)

        # Sort the indices to get the original order
        sorted_indices = torch.argsort(indices, stable=True)

        # Calculate the cumulative sum of the counts
        cum_sum = counts.cumsum(0)

        # Add 0 as the first element and remove the last element
        cum_sum = torch.cat((torch.tensor([0], device=embeddings.device), cum_sum[:-1]))

        # Get the unique indices by using the cumulative counts
        unique_indices = sorted_indices[cum_sum]

        # Sort the unique indices and apply them to the embeddings and labels
        unique_indices, _ = torch.sort(unique_indices, stable=True)

        return embeddings[unique_indices], labels[unique_indices], unique_indices

    @staticmethod
    def _calculate_similarity_scores(embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates the similarity scores between all embeddings.

        Args:
            embeddings (torch.Tensor): Tensor of embeddings.

        Returns:
            torch.Tensor: Matrix of similarity scores.
        """
        # Normalize the embeddings
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)

        # Calculate the similarity scores
        similarity_score = torch.matmul(embeddings_normalized, torch.t(embeddings_normalized))

        # Set the diagonal elements to -1 as it is the product of the same embedding
        similarity_score.fill_diagonal_(-1)

        return similarity_score

    @staticmethod
    def _create_positive_and_negative_mask(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a mask for positives and negatives based on labels.

        Args:
            labels (torch.Tensor): Tensor of labels.

        Returns:
            Tuple: Positive and negative mask.
        """
        # Create a mask for elements with the same label
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        # Clone the label mask to create the positive mask
        positive_mask = label_mask.detach().clone()

        # Set the diagonal elements to false as it is the product of the same embedding
        positive_mask.fill_diagonal_(False)

        # Create the negative mask by inverting the label mask
        negative_mask = ~label_mask

        return positive_mask, negative_mask

    def _mine_positives(
            self,
            scores: torch.Tensor,
            positive_mask: torch.Tensor,
            negative_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mines positive samples based on the specified positive mining strategy.

        Args:
            scores (torch.Tensor): Matrix of similarity scores.
            positive_mask (torch.Tensor): Mask for positives.
            negative_mask (torch.Tensor): Mask for negatives.

        Returns:
            Tuple: Scores and indices of the mined positive samples and a mask to filter out invalid scores.
        """
        # Clone scores to avoid modifying the original tensor
        positive_scores = scores.detach().clone()

        # Mask out negative samples
        positive_scores[negative_mask] = -1

        # Mask out matches between the same elements
        positive_scores[positive_scores > 0.9999] = -1

        # Easy-positive mining
        if self.positive_mining_strategy == 'easy':
            result, indices = torch.max(positive_scores, dim=1)

        # Random positive mining
        else:
            # Get the first index of the elements in the positive mask
            indices = torch.argmax(positive_mask.int(), dim=1)

            # Get the corresponding similarity score for each index
            result = positive_scores[torch.arange(indices.size(0)), indices]

            # If none of the elements is true set the corresponding score and the index to -1
            any_true = positive_mask.any(dim=1)
            indices[~any_true] = -1
            result[~any_true] = -1

        # Mask to filter out invalid scores
        valid = (result > -1) & (result < 1)

        # Update positive scores
        result = scores[torch.arange(0, scores.size(0)), indices]

        return result, indices, valid

    def _mine_negatives(
            self,
            scores: torch.Tensor,
            positive_scores: torch.Tensor,
            positive_mask: torch.Tensor,
            negative_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mines negative samples based on the specified negative mining strategy.

        Args:
            scores (torch.Tensor): Matrix of similarity scores.
            positive_scores (torch.Tensor): Scores of the mined positive samples.
            positive_mask (torch.Tensor): Mask for positives.
            negative_mask (torch.Tensor): Mask for negatives.

        Returns:
            Tuple: Scores and indices of the mined negative samples and a mask to filter out invalid scores.
        """
        # Clone scores to avoid modifying the original tensor
        negative_scores = scores.detach().clone()

        # Mask out positive samples
        negative_scores[positive_mask] = -1

        # Hard-negative mining
        if self.negative_mining_strategy == 'hard':
            result, indices = torch.max(negative_scores, dim=1)

        # Semi-hard-negative mining
        elif self.negative_mining_strategy == 'semi-hard':
            # Semi-hard-negative criteria
            semi_hard_mask = (negative_scores < positive_scores) & (negative_scores + self.margin > positive_scores)

            # Mask out the non semi-hard-negative samples
            negative_scores[~semi_hard_mask] = -1

            # Get the corresponding similarity score and the index
            result, indices = torch.max(negative_scores, dim=1)

        # Random negative mining
        else:
            # Get the first index of the elements in the negative mask
            indices = torch.argmax(negative_mask.int(), dim=1)

            # Get the corresponding similarity score for each index
            result = negative_scores[torch.arange(indices.size(0)), indices]

            # If none of the elements is true set the corresponding score and the index to -1
            any_true = positive_mask.any(dim=1)
            indices[~any_true] = -1
            result[~any_true] = -1

        # Mask to filter out invalid scores
        valid = (result > -1) & (result < 1)

        # Update negative scores
        result = scores[torch.arange(0, scores.size(0)), indices]

        return result, indices, valid

    def mine_triplets(
            self,
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor,
            anchor_label: torch.Tensor,
            positive_label: torch.Tensor,
            negative_label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mines triplets based on the specified mining strategies.

        Args:
            anchor (torch.Tensor): Tensor of anchor embeddings.
            positive (torch.Tensor): Tensor of positive embeddings.
            negative (torch.Tensor): Tensor of negative embeddings.
            anchor_label (torch.Tensor): Tensor of anchor labels.
            positive_label (torch.Tensor): Tensor of positive labels.
            negative_label (torch.Tensor): Tensor of negative labels.

        Returns:
            Tuple: Scores, indices and masks of the mined samples and the unique indices.
        """
        # Concatenate the labels and the embeddings
        embeddings = torch.cat((anchor, positive, negative), 0)
        labels = torch.cat((anchor_label, positive_label, negative_label), 0)

        # Remove duplicate embeddings and the corresponding labels
        embeddings, labels, unique_ind = self._filter_duplicates(embeddings=embeddings, labels=labels)

        # Calculate the similarity scores between all pairs of embeddings
        scores = self._calculate_similarity_scores(embeddings=embeddings)

        # Create masks for positives and negatives
        positive_mask, negative_mask = self._create_positive_and_negative_mask(labels=labels)

        # Mine positive samples based on the specified strategy
        positive_scores, positive_ind, valid_positive = self._mine_positives(
            scores=scores,
            positive_mask=positive_mask,
            negative_mask=negative_mask
        )

        # Mine negative samples based on the specified strategy
        negative_scores, negative_ind, valid_negative = self._mine_negatives(
            scores=scores,
            positive_scores=positive_scores,
            positive_mask=positive_mask,
            negative_mask=negative_mask
        )

        return positive_scores, negative_scores, positive_ind, negative_ind, valid_positive, valid_negative, unique_ind
