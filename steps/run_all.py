from step1_load_data import step1_load
from step2_normalize_data import step2_preprocess
from step3_machinelearning import step3_models
from step4_comparation import step4_compare


def main():
    tabela_original = step1_load()
    data = step2_preprocess(tabela_original)
    models = step3_models(data)
    step4_compare(data, models)


if __name__ == "__main__":
    main()
