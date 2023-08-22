from flask import Flask, request, render_template
from inference import get_category, plot_category, tflite_detect_images
from datetime import datetime

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
    # POST method to post the results file
        # Read file from upload
        img = request.files['file']
        print('img-obj')
        print(img)
        # Get category of prediction
        #image_category = get_category(img)
        # Plot the category
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        img_url = plot_category(img, current_time)
        image_category,percentage, saveLabledUrl = tflite_detect_images(img_url)
        showLabel = True
        if(showLabel == True):
            img_url = saveLabledUrl
        # Render the result template
        return render_template('result.html', len = len(image_category), result=image_category, percentage=percentage, img_url=img_url)
    # For GET requests, load the index file
    return render_template('index.html')

@app.route('/reload')
def reload_page():
    # Reindirizza l'utente alla stessa pagina ('index' nel nostro caso)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)