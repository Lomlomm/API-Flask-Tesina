from flask import Flask
import services.processed_data as ProcessData

app = Flask(__name__)


@app.route('/home', methods=['GET'])
def home():
    return 'HELLO'

@app.route('/processData', methods=['GET'])
def getProcessData():
    return ProcessData.getProcessData()

if __name__ == '__main__':
    app.run(debug=True)
