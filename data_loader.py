import numpy as np 
import pandas as pd 
import logging 
import pathlib 
import os 
import argparse
from typing import Optional, Tuple 


from sklearn.model_selection import train_test_split


def load_to_numpy(data_path: str) -> np.ndarray:
    """
    Load data from a CSV file into a numpy array.
    
    Args:
        data_path (str): Path to the CSV file.
        
    Returns:
        np.ndarray: Data loaded into a numpy array.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    df = pd.read_csv(data_path)
    logging.info(f"Data loaded from {data_path} with shape {df.shape}.")
    
    assert isinstance(df, pd.DataFrame), "Data should be a pandas DataFrame."
    assert len(df.shape) == 2, "Data should be a 2D DataFrame."

    return df.to_numpy()


def split_to_sequences(data: np.ndarray, 
                       sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the data into sequences of a specified length.
    
    Args:
        data (np.ndarray): Input data array.
        sequence_length (int): Length of each sequence.
        
    Returns:
        np.ndarray: Array of sequences.
    """
    
    if data.shape[0] % sequence_length != 0:
        nearest_length = (data.shape[0] // sequence_length) * sequence_length 
        logging.warning(
            f"Data length {data.shape[0]} is not a multiple of sequence length {sequence_length}. Truncating to {nearest_length}."
        )
        data = data[:nearest_length, ...]

    # Split the data into sequences
    num_sequences = data.shape[0] // sequence_length

    data_to_label =  data.reshape(num_sequences, sequence_length, -1)

    logging.info(f"Data split into {num_sequences} sequences of length {sequence_length}. Realigning labels to retain pure sequences.")

    clean_label_data = []

    for seq in range(data_to_label.shape[0]):
        labels = data_to_label[seq, :, -1]
        if len(np.unique(labels)) == 1:
            clean_label_data.append(data_to_label[seq, :, :])

    clean_label_data = np.array(clean_label_data)
    data = clean_label_data[:,:,1:-1]
    labels = clean_label_data[:,:, -1].astype(np.int32)

    return data, labels


def temporary_flattening(
                        data: Tuple[np.ndarray, np.ndarray]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optionally use this method to flatten the data for tokenization later on. 
    This method will be replaced by a multivariate tokenizer in the future.
    """
    data, labels = data 

    if len(data.shape) != 3:
        raise ValueError("Data should have size (num_seq, seq_len, num_features).")

    data = data.reshape(data.shape[0] * data.shape[1], -1)
    labels = labels.reshape(labels.shape[0] * labels.shape[1], )

    return data, labels


def main(path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:

    parser = argparse.ArgumentParser()

    parser.add_argument("--seq_len", type= int, 
                        default= None, 
                        help = "Sequence length parameter...")
    parser.add_argument("--path", type = str,
                         default = None,
                         help = "Specify a data path...")
    
    args = parser.parse_args()

    path = args.path 
    seq_len = args.seq_len


    """
    Main function to execute the data loading and preprocessing.

    Args:
        path (Optional[str]): Path to the data directory.
    """

    if path is None:
        path = pathlib.Path(__file__).parent / "data"

    X, y = [], []

    for file in os.listdir(path):
        if file.endswith(".csv"):
            data_path = os.path.join(path, file)
            data = load_to_numpy(data_path)
            if seq_len is not None:
                sequences, labels = split_to_sequences(data, 
                                    sequence_length= seq_len)
                sequences, labels = temporary_flattening((sequences, labels))
                logging.info(f"Processed {file}: Sequences shape {sequences.shape}, Labels shape {labels.shape}.")
                X.append(sequences)
                y.append(labels)
            else:
                sequences, labels = split_to_sequences(data)
                sequences, labels = temporary_flattening((sequences, labels))
                logging.info(f"Processed {file}: Sequences shape {sequences.shape}, Labels shape {labels.shape}.")
                X.append(sequences)
                y.append(labels)

    X = np.concatenate(X, axis = 0)
    y = np.concatenate(y, axis = 0)

    return X, y 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting data loader module...")

    X, y = main()

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 123)

    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    logging.info("All data loading tasks complete ...")
