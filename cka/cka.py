import torch
import sys

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
    # return x.dot(x.T)
    return torch.matmul(x,x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
    dot_products = x.dot(x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
    if not torch.allclose(gram, gram.T):
        raise ValueError('Itorchut must be a symmetric matrix.')
    # gram = gram.copy()
    # gram = gram.clone().detach()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        torch.fill_diagonal(gram, 0)
        means = torch.sum(gram, 0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        torch.fill_diagonal(gram, 0)
    else:
        means = torch.mean(gram, 0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def CKA(gram_x, gram_y, debiased=False):
    """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    # scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    scaled_hsic = gram_x.view([-1]).dot(gram_y.view([-1]))
    # scaled_hsic = torch.dot(gram_x.view([-1]),gram_y.view([-1]))

    normalization_x = torch.linalg.norm(gram_x)
    normalization_y = torch.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
            xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


# def feature_space_linear_cka(features_x, features_y, debiased=False):
#     """Compute CKA with a linear kernel, in feature space.
#
#   This is typically faster than computing the Gram matrix when there are fewer
#   features than examples.
#
#   Args:
#     features_x: A num_examples x num_features matrix of features.
#     features_y: A num_examples x num_features matrix of features.
#     debiased: Use unbiased estimator of dot product similarity. CKA may still be
#       biased. Note that this estimator may be negative.
#
#   Returns:
#     The value of CKA between X and Y.
#   """
#     features_x = features_x - torch.mean(features_x, 0, keepdim=True)
#     features_y = features_y - torch.mean(features_y, 0, keepdim=True)
#
#     a = torch.mm(features_x.t(), features_y)
#     b = torch.mm(features_x.t(), features_x)
#     c = torch.mm(features_y.t(), features_y)
#     dot_product_similarity = torch.linalg.norm(a) ** 2
#     normalization_x = torch.linalg.norm(b)
#     normalization_y = torch.linalg.norm(c)
#
#     if debiased:
#         n = features_x.shape[0]
#         # Equivalent to torch.sum(features_x ** 2, 1) but avoids an intermediate array.
#         sum_squared_rows_x = torch.einsum('ij,ij->i', features_x, features_x)
#         sum_squared_rows_y = torch.einsum('ij,ij->i', features_y, features_y)
#         squared_norm_x = torch.sum(sum_squared_rows_x)
#         squared_norm_y = torch.sum(sum_squared_rows_y)
#
#         dot_product_similarity = _debiased_dot_product_similarity_helper(
#             dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
#             squared_norm_x, squared_norm_y, n)
#         normalization_x = torch.sqrt(_debiased_dot_product_similarity_helper(
#             normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
#             squared_norm_x, squared_norm_x, n))
#         normalization_y = torch.sqrt(_debiased_dot_product_similarity_helper(
#             normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
#             squared_norm_y, squared_norm_y, n))
#
#     return dot_product_similarity / (normalization_x * normalization_y)
def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - torch.mean(features_x, 0, keepdims=True)
  features_y = features_y - torch.mean(features_y, 0, keepdims=True)

  dot_product_similarity = torch.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = torch.linalg.norm(features_x.T.dot(features_x))
  normalization_y = torch.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to torch.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = torch.sum(features_x ** 2, 1)
    sum_squared_rows_y = torch.sum(features_y ** 2, 1)
    squared_norm_x = torch.sum(sum_squared_rows_x)
    squared_norm_y = torch.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = torch.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = torch.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)