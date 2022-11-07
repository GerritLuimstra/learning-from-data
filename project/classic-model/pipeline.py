import argparse
import os
import time

train_file = "../data/train.tsv"
dev_file = "../data/dev.tsv"
test_file = "../data/test.tsv"

def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str,
                        default=time.strftime("%Y%m%d-%H%M%S"),
                        help="The name for this experiment")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="The number of runs for this experiment")
    parser.add_argument("-t", "--test", default=False, action="store_true",
                        help="Whether to predict and evaluate the test set")
    parser.add_argument("-ow", "--overwrite", default=False,
                        action="store_true", help="Whether to overwrite \
                        previous results")
    parser.add_argument("-m", "--model_name", type=str, default='nb', help="The model to use. Can be one of ['nb', 'et', 'rf', 'knn', 'svc', 'linearsvc']")
    parser.add_argument("-a", "--args", default=[], nargs='+', help="The arguments passed to the ML model")
    args = parser.parse_args()
    return args

def parse_values(values):
    """Parses the values of the classifier"""
    values_ = []
    for value in values:
        if ":" in value:
            values_.append(value.replace(":", ""))
        elif "." in value:
            values_.append(float(value))
        elif value == "None":
            values_.append(None)
        else:
            values_.append(int(value))
    return values_

def get_model_parameters(args):
    """Extract the model parameters"""
    # Check if the classifier arguments are properly given
    if not len(args.args) % 2 == 0:
        print("Invalid arguments specified. Should be in the form: param1 value1 param2 value2")
        exit(0)

    # Parse the arguments
    params = args.args[0::2]
    values = parse_values(args.args[1::2])
    param_dict = dict(zip(params, values))

    return param_dict

def get_model_args(args, directory, run):
    """Get command line arguments for train.py."""
    train_args = ""

    train_args += f"-tf {train_file} "
    train_args += f"-mf {directory}/models/{run} "
    train_args += f"-ff {directory}/models/{run}-features "
    train_args += f"-s {run} "
    train_args += f"-m {args.model_name} "
    
    # Add model parameters if supplied
    if args.args:
        sep = " "
        train_args += f"-a {sep.join(args.args)}"

    return train_args

def train(args, directory, run):
    """Run train.py with the specified arguments and in the given experiment
    directory."""

    # Skip if we already have a trained model and are not overwriting.
    previous_exists = os.path.exists(f"{directory}/models/{run}")
    if not args.overwrite and previous_exists:
        return

    # Train a new model.
    args = get_model_args(args, directory, run)
    os.system(f"python3 train.py {args}")

def get_predict_args(directory, run, test):
    """Get command line arguments for predict.py."""

    predict_args = ""

    if test:
        predict_args += f"-if {test_file} "
        predict_args += f"-of {directory}/test-out/{run}.out "
    else:
        predict_args += f"-if {dev_file} "
        predict_args += f"-of {directory}/out/{run}.out "

    predict_args += f"-mf {directory}/models/{run} "

    return predict_args

def predict(args, directory, run):
    """Run predict.py with the specified arguments and in the given experiment
    directory."""

    # Only predict if there are no predictions yet or we are overwriting.
    previous_exists = os.path.exists(f"{directory}/out/{run}.out")
    if args.overwrite or not previous_exists:
        # Get predictions on dev.
        predict_args = get_predict_args(directory, run, False)
        os.system(f"python3 predict.py {predict_args}")

    # Skip predicting on test set if not requested.
    if not args.test:
        return

    # Only predict if there are no predictions yet or we are overwriting.
    previous_exists = os.path.exists(f"{directory}/test-out/{run}.out")
    if args.overwrite or not previous_exists:
        # Get predictions on test.
        predict_args = get_predict_args(directory, run, True)
        os.system(f"python3 predict.py {predict_args}")

def get_evaluate_args(directory, test):
    """Get command line arguments for evaluate.py."""

    evaluate_args = ""

    if test:
        evaluate_args += f"-tf {test_file} "
        evaluate_args += f"-of {directory}/test_scores.txt "
        evaluate_args += f"-pd {directory}/test-out "
    else:
        evaluate_args += f"-tf {dev_file} "
        evaluate_args += f"-of {directory}/scores.txt "
        evaluate_args += f"-pd {directory}/out "

    return evaluate_args

def evaluate(args, directory):
    """Run evaluate.py with the specified arguments and in the given experiment
    directory."""

    # Get scores on dev.
    evaluate_args = get_evaluate_args(directory, False)
    os.system(f"python3 evaluate.py {evaluate_args}")

    # Skip evaluating on test set if not requested.
    if not args.test:
        return

    # Get scores on test.
    evaluate_args = get_evaluate_args(directory, True)
    os.system(f"python3 evaluate.py {evaluate_args}")

def main():
    """Main function to run the pipeline."""

    # Read the command line arguments.
    args = create_arg_parser()

    # Create required directories.
    experiments_directory = f"results/{args.name}"
    os.makedirs(experiments_directory, exist_ok=True)
    os.makedirs(f"{experiments_directory}/models", exist_ok=True)
    os.makedirs(f"{experiments_directory}/out", exist_ok=True)
    os.makedirs(f"{experiments_directory}/test-out", exist_ok=True)

    # Run the pipeline a given number of times.
    for run in range(args.runs):
        train(args, experiments_directory, run)
        predict(args, experiments_directory, run)

    # Evaluate all runs.
    evaluate(args, experiments_directory)


if __name__ == "__main__":
    main()
