# a simple evaluation metric by Bo Xin

# k:          num of labels
# label:      ground truth label matrix (numpy ndarray), nxk where n is the num of sampels per batch. 
#             The value of a label belongs to [0,K-1] where K is the largest label index.
# pred_prob:  predicted probability matrix (numpy ndarray), nxK where n is the num of sampels per batch.
  
# Accuracy1:  check if the ground truth k labels are exactly same with the predicted top k labels.  
# Accuracy2:  check how much percent of the ground truth k labels were correctly found in the predicted top k labels.

  def Accuracy1(label, pred_prob):
      pred = np.argpartition(pred_prob, -k, axis=1)[:,-k:]
      t_score = np.zeros(label.shape)
      for i in range(k):
          for j in range(k):
              t_score[:,i] = t_score[:,i]+(label[:,i]==pred[:,j])
      return np.sum((np.sum(t_score, axis=1)==k))*1.0/(pred.shape[0])
      
  def Accuracy2(label, pred_prob):
      pred = np.argpartition(pred_prob, -k, axis=1)[:,-k:]
      t_score = np.zeros(label.shape)
      for i in range(k):
          for j in range(k):
              t_score[:,i] = t_score[:,i]+(label[:,i]==pred[:,j])
      return np.sum(np.sum(t_score))*1.0/(pred.shape[0]*k)

