from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import dash
from dash import html
from dash import  dcc
from dash.dependencies import Input, Output, State
import base64
from PIL import Image
from clip_model import CLIP
import index_image_database as img_db
import kd_tree as kd
import database_connection as db

server = Flask(__name__)
app = dash.Dash(__name__, server=server)
kd_tree = kd.deserialize_tree(db.connect('imageKDTree').find_one())

# clip_model=CLIP()
# clip_model.load_model()

def get_image_caption_pair(user_input_text):
    return img_db.find_most_similar_image(user_input_text, kd_tree)

app.layout = html.Div(
    children=[
        dcc.Location(id='url', refresh=False),
        html.H1('Image search 🖼️🔍', style={'text-align': 'center', 'margin-bottom': '20px'}),
        html.Div(
            children=[
                html.H2('Search for an image:', style={'margin-bottom': '10px'}),
                dcc.Input(
                    id='text-input',
                    placeholder='Describe image to search for...',
                    type='text',
                    style={'width': '480px', 'border': '1px solid #ccc', 'padding': '10px'}
                ),
                html.Button('Generate Image', id='generate-image-button', n_clicks=0, style={'margin-top': '10px', 'width': '100%', 'font-size': '16px'}),
                html.Div(id='output-text-input', style={'margin-top': '20px'}),
            ],
            style={'max-width': '500px', 'margin': '0 auto'}
        ),
    ],
    style={'background-color': '#333', 'color': 'white', 'font-family': 'Arial, sans-serif','width':'100%','height':'100%'}
)

@app.callback(
    Output('output-text-input', 'children'),
    [Input('generate-image-button', 'n_clicks')],
    [State('text-input', 'value')]
)
def generate_image(n_clicks, text):
    if n_clicks > 0:
        print("text is:", text)
        image_data, caption = get_image_caption_pair(text)
        return html.Div(
            children=[
                html.H4('Retrieved Image:'),
                html.Img(
                    src='data:image/png;base64,{}'.format(image_data),
                    style={'width': '300px', 'height': '300px', 'margin-bottom': '10px'}
                ),
                html.Div("Image caption:", style={'font-weight': 'bold'}),
                html.Div(caption)
            ],
            style={'text-align': 'center'}
        )


if __name__ == '__main__':
    # host="0.0.0.0"
    app.run_server(host="0.0.0.0",port=8050, debug=True)