from flask import Flask, render_template, redirect, request

import caption_module

#__name__ = __main__
app = Flask(__name__)

# model = joblib.load()

@app.route("/")
def hello():
	return render_template("index.html")

@app.route("/",methods =['POST'])
def get_image():

	if request.method == 'POST':
		f = request.files['userfile']
		print(f.filename)
		path = f"./static/{f.filename}"
		f.save(path)

		caption = caption_module.caption_this(path)
		
		result_dict = {
			'image':path,
			'caption':caption
		}

	return render_template("index.html", result = result_dict)



# @app.route('/submit', methods =['POST'])
# def submit_data():
# 	if request.method == 'POST':
# 		name = request.form['username']

# 		f = request.files['userfile']
# 		print(f)
# 		f.save(f.filename)

	# return f"<h3>Hello {name}</h3>"

 

if __name__ == '__main__':
	# app.debug = True
	app.run(debug = True)