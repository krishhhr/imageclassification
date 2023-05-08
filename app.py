from flask import Flask, request, render_template
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import firebase_admin
from firebase_admin import credentials, firestore
import json

# Initialize Firebase
cred = credentials.Certificate("./iacv-4b46e-firebase-adminsdk-irs0w-1944ded158.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

# Initialize Clarifai channel and stub
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

metadata = (('authorization', 'Key ' + 'd819e105d26e46e0858e6fa1a7b47bfc'),)


@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        image_url = request.form['image_url']

        # Make the API call to classify the image
        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=resources_pb2.UserAppIDSet(user_id='clarifai', app_id='main'),
                model_id='general-image-recognition',
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                url=image_url
                            )
                        )
                    )
                ]
            ),
            metadata=metadata
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            return "Failed to classify image. Please try again."

        # Retrieve the predicted concepts
        output = post_model_outputs_response.outputs[0]
        predicted_concepts = [(concept.name, concept.value) for concept in output.data.concepts]
        predicted_concepts_json = json.dumps(predicted_concepts)

        # Save the image and predictions in Firebase
        doc_ref = db.collection('images').document()
        doc_ref.set({
            'image_url': image_url,
            'predictions': predicted_concepts_json
        })

        return render_template('result.html', image_url=image_url, concepts=predicted_concepts)

    return render_template('index.html')


@app.route('/gallery', methods=['GET'])
def search_images():
    # Query Firestore for images
    images = db.collection('images').stream()

    all_images = []
    for image in images:
        image_data = image.to_dict()
        all_images.append(image_data['image_url'])

    return render_template('search.html', images=all_images)


@app.route('/images', methods=['GET', 'POST'])
def show_images():
    if request.method == 'POST':
        concept = request.form['concept']
        images = db.collection('images').stream()

        image_urls = []
        for image in images:
            image_data = image.to_dict()
            predicted_concepts = image_data.get('predictions')
            if predicted_concepts:
                concept_values = json.loads(predicted_concepts)
                for concept_value in concept_values:
                    if isinstance(concept_value, list) and concept_value[0].lower() == concept.lower():
                        image_urls.append(image_data['image_url'])
                        break

        if not image_urls:
            message = f"No images found for '{concept}'."
            return render_template('images.html', concept=concept, image_urls=image_urls, message=message)

        return render_template('images.html', concept=concept, image_urls=image_urls)

    return render_template('images.html')


if __name__ == '__main__':
    app.run(debug=True)


