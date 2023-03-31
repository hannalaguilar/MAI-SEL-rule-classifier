from typing import List, Optional
import time
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

from src.cn2 import run_cn2

folder_path = Path(__file__).parent.parent


@dataclass
class Dataset:
    name: str
    continuous_attributes: Optional[List[str]]


if __name__ == "__main__":
    # Three datasets
    iris_dataset = Dataset('iris',
                           ['sepal_length',
                            'sepal_width',
                            'petal_length',
                            'petal_width'])
    titanic_dataset = Dataset('titanic', ['Age', 'Fare'])
    obesity_dataset = Dataset('obesity',
                              ['Age', 'Height', 'Weight',
                               'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'])

    for dataset in [obesity_dataset]:
        data_dir = folder_path / 'data'
        print('{:-^85}'.format(dataset.name.upper()))
        data_path = data_dir / f'{dataset.name}.csv'
        start_time = time.time()
        rule_list, train_acc, test_acc = run_cn2(data_path,
                                                 dataset.continuous_attributes,
                                                 verbose=True)
        elapse_time = time.time() - start_time
        print(f'Train accuracy: {train_acc:.2f}')
        print(f'Test accuracy: {test_acc:.2f}')
        print(f'Training time: {elapse_time:.4f}s')

        # save rules in csv
        pd.DataFrame(rule_list,
                     columns=['rule', 'precision', 'recall']). \
            to_csv(data_dir / f'{dataset.name}_rules.csv')
        print('')
