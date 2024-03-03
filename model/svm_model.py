from convert_json_to_pd import Convert2DF

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    df_data = Convert2DF()
    print(df_data.shape)
    
main()