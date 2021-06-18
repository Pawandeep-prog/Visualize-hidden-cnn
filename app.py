from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from keras.models import load_model, Model
import matplotlib.pyplot as plt
from tqdm import tqdm

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

#layers = []
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/root')
def root():
	return render_template("root.html")

@app.route('/choose_img', methods=['GET'])
def choose_img():
	global layers, dl_model

	layers = []

	model_path = request.args['path']
	dl_model = load_model(model_path)

	for i in range(len(dl_model.layers)):
		if len(dl_model.layers[i].output_shape) == 4:
			layers.append(dict(id=i, i=dl_model.layers[i].name))

	print(layers)

	print("="*50)
	print("model loaded")

	return render_template("choose_img.html")

@app.route('/model', methods=['GET', 'POST'])
def model():

	print(dl_model.layers)

	img = request.files['img']
	img.save('static/img.jpg')

	img = cv2.imread('static/img.jpg', 0)

	img = cv2.resize(img, (dl_model.layers[0].input_shape[1], dl_model.layers[0].input_shape[2]))
	img = img.reshape(1, dl_model.layers[0].input_shape[1], dl_model.layers[0].input_shape[2], 1)
	img = img/255.0

	for k in tqdm(range(len(layers))):
		temp_model = Model(dl_model.layers[0].input, dl_model.layers[k].output)

		if len(temp_model.layers[-1].output_shape) == 4:
			op_images = temp_model.layers[-1].output_shape[-1]

			pred = temp_model.predict(img)

			cnt = 0
			fig, axes = plt.subplots(int(op_images**0.5), int(op_images**0.5), figsize=(15,15))
			for i in range(int(op_images**0.5)):
				for j in range(int(op_images**0.5)):
					axes[i, j].imshow(pred[0, :, :, cnt], cmap='gray')
					axes[i,j].axis('off')
					
					cnt += 1

			cnt = 0
			plt.savefig(f"static/{k}.png", bbox_inches='tight',pad_inches = 0)
	
	

	return render_template("model.html", layers=layers)

@app.route('/inspect')
def inspect():

	x = request.args['v']+'.png'

	print("="*50)
	print(x)

	return render_template('inspect.html', data=str(x), ly = request.args['v'], layer_name=layers[int(request.args['v'])])


if __name__ == "__main__":
	app.run(debug=True)

