import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_class = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):

    scores = np.dot(X[i], W)
    scores -= np.max(scores)

    current_class_score = scores[y[i]]

    # loss += - np.log(np.exp(current_class_score) / np.sum(np.exp(scores)))
    loss += - current_class_score + np.log(np.sum(np.exp(scores)))

    for j in range(num_class):

      dW[:, j] += np.dot(np.exp(scores[j]), X[i].T) / np.sum(np.exp(scores))

      if j == y[i]:
        dW[:, y[i]] -= X[i].T

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # loss
  num_train = X.shape[0]
  scores = X.dot(W)   # 200 x 3073 * 3073 x 10 = 200 x 10
  # scores -= np.transpose([np.max(scores, axis=1)])
  scores -= np.max(scores, axis=1)[:, np.newaxis]

  scores_label = scores[[range(num_train)], y] # 1 x 200
  loss_array = - scores_label + np.log(np.sum(np.exp(scores), axis=1)) # 200 x 10

  loss = np.sum(loss_array) / num_train + reg * np.sum(W * W)

  # dW
  scores_probs = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:, np.newaxis]
  scores_probs[range(num_train), y] -= 1
  dW = np.dot(X.T, scores_probs)

  dW /= num_train
  dW += 2 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

