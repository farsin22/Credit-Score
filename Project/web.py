from flask import Flask , render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('farsin.html')
    return redirect(url_for('predict'))
@app.route('/predict',methods=['POST'])
def predict():
    credit=[float(x)for x in request.form.values()]
    final_credit=[np.array(credit)]
    output=model.predict(final_credit)
    return render_template('result.html',prediction_result='{}'.format(output))

if __name__=='__main__':
    app.run(debug=True)

