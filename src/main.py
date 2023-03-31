from src.cn2 import run_cn2


if __name__ == "__main__":
    DATA_PATH = '../data/titanic.csv'
    RULE_LIST = run_cn2(DATA_PATH, verbose=True)
