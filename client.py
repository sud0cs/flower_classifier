import requests as r
import os
def main():
    # L'aplicacio de flask tambe accepta arguments a les URLs:
    # x = r.get('http://127.0.0.1:5000/api/predict/svcmodel?petal_width=2.2&petal_length=2.2')
    # print(x.text)
    for model in os.listdir('models'):

        data = {
            'petal_width': 2.2,
            'petal_length': 2.2
        }
        x = r.post(f'http://127.0.0.1:5000/api/predict/{model.replace(".pck", "")}', json=data)
        print(x.text)
if __name__ == '__main__':
    main()
