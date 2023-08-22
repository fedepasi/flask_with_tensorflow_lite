import matplotlib
import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.lite.python.interpreter import Interpreter

# https: // stackoverflow.com/questions/53684971/assertion-failed-flask-server-stops-after-script-is-run
matplotlib.use('Agg')

def get_category(img):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file with 3 color channels

    Returns:
        [str]: Prediction
    """
    # Read an image from a file into a numpy array
    img = mpimg.imread(img)
    # Convert to float32
    img = tf.cast(img, tf.float32)
    # Resize to 320x320 (size the model is expecting)
    img = tf.image.resize(img, [320, 320])
    # Expand img dimensions from (224, 224, 3) to (1, 224, 224, 3) for set_tensor method call
    img = np.expand_dims(img, axis=0)

    tflite_model_file = 'static/model/f1_2023_detect_lite.tflite'

    with open(tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    prediction = []
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    prediction.append(interpreter.get_tensor(output_index))
    print(prediction)
    predicted_label = np.argmax(prediction)
    print('prediction_label')
    print(predicted_label)
    class_names = ['A325-10',
        'A325-31',
        'AMR23-14',
        'AMR23-18',
        'AT04-3',
        'AT04-22',
        'C43-24',
        'C43-77',
        'FW45-2',
        'FW45-23',
        'MCL60-4',
        'MCL60-81',
        'RB19-1',
        'RB19-11',
        'SF23-16',
        'SF23-55',
        'VF23-20',
        'VF23-27',
        'W14-44',
        'W14-63',
        'f1-14',
        'f1-14-side',
        'f1-18',
        'f1-18-side',
        'f1-1',
        'f1-1-side',
        'SafetyCar-Aston',
        'AT04-21',
        'RB-18-1']
    
    return class_names[predicted_label]

# def plot_category(img):
def plot_category(img, current_time):
    """Plot the input image. Timestamp used to help Flask grab the correct image.

    Args:
        img [jpg]: image file
        current_time: timestamp
    """
    # Read an image from a file into a numpy array
    img_new_url =f'static/images-v2/output_{current_time}.jpg';
    img.save(img_new_url)
    # img = mpimg.imread(img)
    # # Remove the plotting ticks
    # plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(img, cmap=plt.cm.binary)
    # # To make sure Flask grabs the correct image to plot
    # strFile = f'static/images/output_{current_time}.png'
    # if os.path.isfile(strFile):
    #     os.remove(strFile)
    # # Save the image with the file name that result.html is using as its img src
    # plt.savefig(strFile)
    return img_new_url

def substrF1(str,per):
    str.replace("A325-10", "label_")
    str.replace("A325-31", "label_")
    str.replace("AMR23-14", "label_")
    str.replace("AMR23-18", "label_")
    str.replace("AT04-3", "label_")
    str.replace("AT04-22", "label_")
    str.replace("C43-24", "label_")
    str.replace("C43-77", "label_")
    str.replace("FW45-2", "label_")
    str.replace("FW45-23", "label_")
    str.replace("MCL60-4", "label_")
    str.replace("MCL60-81", "label_")
    str.replace("RB19-1", "label_")
    str.replace("RB19-11", "label_")
    str.replace("SF23-16", "label_")
    str.replace("SF23-55", "label_")
    str.replace("VF23-20", "label_")
    str.replace("VF23-27", "label_")
    str.replace("W14-44", "label_")
    str.replace("W14-63", "label_")
    str.replace("SafetyCar-Aston", "label_")
    str.replace("AT04-21", "label_")
    str.replace("RB-18-1", "label_")
    return str

def tflite_detect_images(imgurl, min_conf=0.5, savepath='static/txt_results'):
    # # Grab filenames of all images in test folder
    # images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')
    # print('images')
    # print(images)
    modelpath = 'static/model/f1_2023_detect_lite.tflite'
    lblpath = 'static/model/labelmap.txt'

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Loop over every image and perform detection
    #for image_path in images:
    print('image_path2')
    print(imgurl)
    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(imgurl)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255, 255, 255), 2) #blue (179, 105, 0)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])


    # All the results have been drawn on the image, now display the image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12,16))
    plt.imshow(image)
    # Remove the plotting ticks
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    # Salva l'immagine su disco
    # Imposta i margini del plot in modo che non ci sia spazio bianco intorno all'immagine
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    saveLabledUrl ='static/images-with-label/'+os.path.basename(imgurl)
    saveLabledUrl = saveLabledUrl.replace("output_", "label_")
    plt.savefig(saveLabledUrl, bbox_inches='tight', pad_inches=0)
    # Get filenames and paths
    image_fn = os.path.basename(imgurl)
    base_fn, ext = os.path.splitext(image_fn)
    txt_result_fn = base_fn +'.txt'
    txt_savepath = os.path.join(savepath, txt_result_fn)

    # Write results to text file
    # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
    txt_result_string = []
    percentage = []
    with open(txt_savepath,'w') as f:
        for detection in detections:
            txt_result_string.append(detection[0])
            percentage.append(round(detection[1]*100))
            f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

    return txt_result_string,percentage, saveLabledUrl

