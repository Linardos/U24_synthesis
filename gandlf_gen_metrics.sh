gandlf generate-metrics \
  # -h, --help         Show help message and exit
  # -v, --version      Show program's version number and exit.
  -c ./experiments/002_experiment_G/gandlf_config.yaml
  -i , --input-data    CSV file that is used to generate the metrics; should contain 3 columns: 'SubjectID,Target,Prediction'
  -o , --output-file   Location to save the output dictionary. If not provided, will print to stdout.