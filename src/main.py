from pathlib import Path

from src.cn2 import run_cn2

folder_path = Path(__file__).parent.parent

if __name__ == "__main__":
    DATASETS = ['titanic']
    for dataset in DATASETS:
        print('{:-^85}'.format(dataset.upper()))
        data_path = folder_path / 'data' / f'{dataset}.csv'
        RULE_LIST = run_cn2(data_path, verbose=True)
        print('')
