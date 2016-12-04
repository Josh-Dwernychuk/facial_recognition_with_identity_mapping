import argparse
import base64
import json
from googleapiclient import discovery
import httplib2
from oauth2client.client import GoogleCredentials
import json
from pprint import pprint
from PIL import Image
from PIL import ImageDraw
from fullcontact import FullContact

def get_vision_service():
    #set up API connection
    credentials = GoogleCredentials.get_application_default()
    return discovery.build('vision', 'v1', credentials=credentials)


def pythagorean(x1,y1,x2,y2):
    #Pythagorean theorum for computing distances between
    #facial feature point pixel coordinates
    a=x2-x1
    b=y2-y1
    return (a**2+b**2)**(.5)


def identify_landmark(photo_file, max_results):
    #Funtion that collects facial feature pixel coordinates from the input image and
    #returns a list of pixel distances between facial features
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)
    with open(photo_file, 'rb') as image:
        image_content = base64.b64encode(image.read())
        service_request = service.images().annotate(body={
            'requests': [{
                'image': {
                    'content': image_content.decode('UTF-8')
                },
                'features': [
                {
                    'type': 'LANDMARK_DETECTION',
                    'maxResults': max_results
                },
                {
                    "type":"FACE_DETECTION",
                    "maxResults":max_results
                }
                ]
            }]
        })

        response = service_request.execute()
        label = response['responses']

        positions_dump=label[0]['faceAnnotations'][0]['landmarks']
        x_list=[]
        y_list=[]
        z_list=[]
        type_list=[]

        for i in range(0, len(positions_dump)):
            x_list.append(positions_dump[i]['position']['x'])
            y_list.append(positions_dump[i]['position']['y'])
            z_list.append(positions_dump[i]['position']['z'])
            type_list.append(positions_dump[i]['type'])

        #FEATURE PIXEL COORDINATE DISTANCES COMPUTEd FROM JSON GATHERED FROM API
        global distance_list
        distance_list=[]
        eye_to_eye=x_list[1]-x_list[0]
        distance_list.append(eye_to_eye)
        left_brow_width= x_list[3]-x_list[2]
        distance_list.append(left_brow_width)
        right_brow_width= x_list[5]-x_list[4]
        distance_list.append(right_brow_width)
        nose_tip_to_left_eye=pythagorean(x_list[1], x_list[7], y_list[1], y_list[7])
        distance_list.append(nose_tip_to_left_eye)
        nose_tip_to_right_eye=pythagorean(x_list[1], x_list[7], y_list[1], y_list[7])
        distance_list.append(nose_tip_to_right_eye)
        nose_tip_to_upper_lip=y_list[8]-y_list[7]
        distance_list.append(nose_tip_to_upper_lip)
        upper_lip_to_lower_lip=y_list[9]-y_list[8]
        distance_list.append(upper_lip_to_lower_lip)
        width_of_mouth=x_list[11]-x_list[10]
        distance_list.append(width_of_mouth)
        left_eye_width=x_list[17]-x_list[19]
        distance_list.append(left_eye_width)
        left_eye_width=x_list[22]-x_list[24]
        distance_list.append(left_eye_width)
        ear_to_ear=x_list[29]-x_list[28]
        distance_list.append(ear_to_ear)
        forehead_to_left_ear=pythagorean(x_list[30], x_list[28], y_list[30],y_list[28])
        distance_list.append(forehead_to_left_ear)
        forehead_to_right_ear=pythagorean(x_list[30], x_list[29], y_list[30],y_list[29])
        distance_list.append(forehead_to_right_ear)
        forehead_to_chin=y_list[30]-y_list[31]
        distance_list.append(forehead_to_chin)
        right_eye_to_chin=pythagorean(x_list[1], x_list[31], y_list[1],y_list[31])
        distance_list.append(right_eye_to_chin)
        left_eye_to_chin=pythagorean(x_list[0], x_list[31], y_list[0],y_list[31])
        distance_list.append(left_eye_to_chin)
        left_ear_to_chin=pythagorean(x_list[28], x_list[31], y_list[28],y_list[31])
        distance_list.append(left_ear_to_chin)
        right_ear_to_chin=pythagorean(x_list[29], x_list[31], y_list[29],y_list[31])
        distance_list.append(right_ear_to_chin)
        nose_bottom_left_to_chin=pythagorean(x_list[15], x_list[31], y_list[15],y_list[31])
        distance_list.append(nose_bottom_left_to_chin)
        nose_bottom_right_to_chin=pythagorean(x_list[14], x_list[31], y_list[14],y_list[31])
        distance_list.append(nose_bottom_right_to_chin)
        chin_width=x_list[33]-x_list[32]
        distance_list.append(chin_width)
        return distance_list


def compare_images(image_1, image_2):
    #Function that compares faces to determine the magnitude of differences
    #between the input image and the pre-cached faces from social media
    find_face(image_1, 'face_img_1.jpg', 4)
    crop_image('face_img_1.jpg','face_img_1_cropped.jpg')
    find_face(image_2, 'face_img_2.jpg', 4)
    crop_image('face_img_2.jpg','face_img_2_cropped.jpg')

    distance_list_1=identify_landmark('face_img_1.jpg', 10)
    distance_list_2=identify_landmark('face_img_2.jpg', 10)
    difference_list=[]
    for i in range(0, len(distance_list_1)):
        difference_list.append(abs(distance_list_2[i]-distance_list_1[i]))
    difference_total=sum(difference_list)
    print 'Face difference score is:'+ str(difference_total)
    global difference_total
    return difference_total


def detect_face(face_file, max_results=4):
    #Function that collects the facial annotations from the API
    image_content = face_file.read()
    batch_request = [{
        'image': {
            'content': base64.b64encode(image_content).decode('utf-8')
            },
        'features': [{
            'type': 'FACE_DETECTION',
            'maxResults': max_results,
            }]
        }]

    service = get_vision_service()
    request = service.images().annotate(body={
        'requests': batch_request,
        })
    response = request.execute()

    return response['responses'][0]['faceAnnotations']

def highlight_faces(image, faces, output_filename):
    #Function that finds the face in the image and puts a box around it for demonstration purposes
    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    for face in faces:
        box = [(v.get('x', 0.0), v.get('y', 0.0))
               for v in face['fdBoundingPoly']['vertices']]
        global vertices
        vertices = face['fdBoundingPoly']['vertices']
        draw.line(box + [box[0]], width=5, fill='#00ff00')

    im.save(output_filename)

def crop_image(image, save):
    #Function that crops the face out of the full image and saves the face as its own image file
    img = Image.open(image)
    w, h = img.size
    img2=img.crop((vertices[3]['x'],vertices[1]['y'],vertices[1]['x'],vertices[3]['y']))
    img2.save(save)

def find_face(input_filename, output_filename, max_results):
    with open(input_filename, 'rb') as image:
        faces = detect_face(image, max_results)
        print('Found {} face{}'.format(
            len(faces), '' if len(faces) == 1 else 's'))

        print('Writing to file {}'.format(output_filename))
        image.seek(0)
        highlight_faces(image, faces, output_filename)

def full_contact_call(handle):
    fc = FullContact('<API KEY>')
    person_profile = fc.person(twitter=handle)
    print person_profile.json()
    print person_profile
    data = person_profile.json()
    return person_profile.json()

value_list=[]
photo_list=[
#List of all pre-cached photos
]
twitter_handle_list=[
#List of pre-cahced twitter handles corresponding to the photos
]

def main(input_photo):
    #Funtion to iterate through pre-cached faces comparing them to the input face image and
    #make a call to full contact to gather personal information for the matching face in the pre-cached data
    for i in photo_list:
        compare_images(input_photo, i)
        value_list.append(difference_total)
    index = value_list.index(min(value_list))
    photo = photo_list[index]
    full_contact_call(twitter_handle_list[index])


#Name input photo as input_photo.jpg
main('input_photo.jpg')
