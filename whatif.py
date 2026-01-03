import gower
import numpy as np

from sklearn import ensemble
from Dataset.dataset import Dataset

def generate_whatif(x_interest, model, dataset) : 
  
  """
  Computes whatif counterfactuals for binary classification models, 
  i.e., the closest data point with a different prediction.
  Parameter: 
    x_interest (np.array with shape (1, num_features)): Datapoint of interest.
    model: Binary classifier which has a predict method.
    dataset (np.array with shape (?, num_features)): Input data
    from which a counterfactual is selected from.

  Returns:
    counterfactual (np.array with shape (1, num_features)): the closest 
    observation/row to x_interest of the input dataset with a different prediction 
    than x_interest. 
  """

  # Prediction of interest
  y_interest = model.predict(x_interest)[0]
  
  # Predictions for the dataset
  # Assuming dataset is X (features), not (X, y)
  y_pred = model.predict(dataset)
  
  # Filter for observations with different prediction
  # We want counterfactual -> different class
  candidates_idx = np.where(y_pred != y_interest)[0]
  
  if len(candidates_idx) == 0:
      return None # No counterfactual found
      
  X_candidates = dataset[candidates_idx]
  
  # Calculate Gower distances
  # gower.gower_matrix(data_x, data_y)
  distances = gower.gower_matrix(x_interest, X_candidates)[0]
  
  # Find index of minimum distance
  min_idx = np.argmin(distances)
  
  # Result is the best candidate
  counterfactual = X_candidates[min_idx].reshape(1, -1)
  
  return counterfactual


def evaluate_counterfactual(counterfactual, x_interest, model) :
  """
   Evaluates if counterfactuals are minimal, i.e., if setting one feature to 
   the value of x_interest still results in a different prediction than for x_interest.
   
   Parameter: 
   counterfactual (np.array with shape (1, num_features)): Counterfactual of `x_interest`. 
   x_interest (np.array with shape (1, num_features)): Datapoint of interest. 
   model: Binary classifier which has a predict method.
  
   Returns: 
   List with indices of features that if set for the counterfactual to the value of 
   `x_interest`, still leads to a different prediction than for x_interest. 
  """
  
  non_minimal_features = []
  
  y_interest = model.predict(x_interest)[0]
  y_cf = model.predict(counterfactual)[0]
  
  if y_interest == y_cf:
      print("Warning: Counterfactual has same prediction as interest point.")
      return []
  
  num_features = x_interest.shape[1]
  
  for i in range(num_features):
      # Create a temporary point where we revert feature i to x_interest value
      temp_cf = counterfactual.copy()
      temp_cf[0, i] = x_interest[0, i]
      
      # Check prediction
      y_temp = model.predict(temp_cf)[0]
      
      # If prediction is still different from x_interest (i.e. same as cf), 
      # then feature i was NOT necessary to change (or at least, reverting it didn't break the flip)
      # Wait, valid counterfactual means prediction != y_interest.
      # If we revert feature i and prediction is STILL != y_interest, it means the change in i was arguably "redundant" 
      # or at least not strictly sufficient on its own to go back to original class?
      # Definition of minimal: changing feature value back to original does NOT flip class back to original?
      # Usually minimal means we changed as few features as possible.
      # If setting feature i back to original value results in prediction == y_cf (still counterfactual state),
      # then this feature's change wasn't the "cause" of the flip?
      # The doc says: "Evaluates if counterfactuals are minimal, i.e., if setting one feature to 
      # the value of x_interest ABSOLUTELY results in a different prediction than for x_interest."
      # Wait, "still leads to a different prediction than for x_interest".
      # So, if we revert feature i, and the prediction is STILL different (still counterfactual), 
      # then feature i isn't *needed* to be changed for the flip?
      # So we return indices of features that "if set ... to value of x_interest, STILL leads to different prediction".
      # This means the change in feature i was NOT necessary for the flip.
      
      if y_temp != y_interest:
          non_minimal_features.append(i)
          
  return non_minimal_features


if __name__ == "__main__":
  
  dataset = Dataset("wheat_seeds", range(0, 7), [7], normalize=True, categorical=True)
  print(dataset.get_input_labels())
  print(dataset.get_output_label())
  print(np.unique(dataset.y, return_counts=True))
    
  # Create a binary classification task
  y = dataset.y
  y[y == 0] = 1
  print(np.unique(dataset.y, return_counts=True))
  
  # Reserve first row of dataset for x_interest, remove from dataset
  X = dataset.X
  x_interest = X[0,:].reshape(1, -1)
  X = np.delete(X, (0), axis = 0)
  y = np.delete(y, (0), axis = 0)
  
  # Fit a random forest to the data
  model = ensemble.RandomForestClassifier(random_state=0)
  model.fit(X, y)
  
  # Probe on x_interest
  print(x_interest)
  print(model.predict(x_interest))
  
  # Compute counterfactual for x_interest
  cf = generate_whatif(x_interest = x_interest, model = model, dataset = X)
  print(cf)
  print(evaluate_counterfactual(counterfactual = cf, x_interest = x_interest, model = model))
  

