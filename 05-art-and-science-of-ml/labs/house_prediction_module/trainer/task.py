import argparse
import os
import json
import shutil

from . import model
    
if __name__ == '__main__' and "get_ipython" not in dir():
  parser = argparse.ArgumentParser()
  # TODO: Add learning_rate and batch_size as command line args
  parser.add_argument(
    '--learning_rate',
    type = float,
    help = 'Set the learning rate for the network training',
    default = 0.01
  )
  parser.add_argument(
    '--batch_size',
    type = int,
    default = 30,
    help = 'Batch size for batch gradient descent'
  )
  parser.add_argument(
      '--output_dir',
      help = 'GCS location to write checkpoints and export models.',
      required = True
  )
  parser.add_argument(
      '--job-dir',
      help = 'this model ignores this field, but it is required by gcloud',
      default = 'junk'
  )
  args = parser.parse_args()
  arguments = args.__dict__
  
  # Unused args provided by service
  arguments.pop('job_dir', None)
  arguments.pop('job-dir', None)
  
  # Append trial_id to path if we are doing hptuning
  # This code can be removed if you are not using hyperparameter tuning
  arguments['output_dir'] = os.path.join(
      arguments['output_dir'],
      json.loads(
          os.environ.get('TF_CONFIG', '{}')
      ).get('task', {}).get('trial', '')
  )
  
  # Run the training
  shutil.rmtree(arguments['output_dir'], ignore_errors=True) # start fresh each time
  
  # Pass the command line arguments to our model's train_and_evaluate function
  model.train_and_evaluate(arguments)