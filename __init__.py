from app.utils.PluginClass import PluginClass
from flask_jwt_extended import jwt_required, get_jwt_identity
from celery import shared_task
from app.utils import DatabaseHandler
from flask import request, send_file
import os
import layoutparser as lp
import cv2
from app.api.records.models import RecordUpdate
from app.api.resources.services import update_cache as update_cache_resources
from app.api.records.services import update_cache as update_cache_records
from PIL import Image, ImageDraw
from bson.objectid import ObjectId
import uuid
from dotenv import load_dotenv
import cv2
load_dotenv()


mongodb = DatabaseHandler.DatabaseHandler()
WEB_FILES_PATH = os.environ.get('WEB_FILES_PATH', '')
ORIGINAL_FILES_PATH = os.environ.get('ORIGINAL_FILES_PATH', '')
USER_FILES_PATH = os.environ.get('USER_FILES_PATH', '')
plugin_path = os.path.dirname(os.path.abspath(__file__))
models_path = plugin_path + '/models'

class ExtendedPluginClass(PluginClass):
    def __init__(self, path, import_name, name, description, version, author, type, settings):
        super().__init__(path, __file__, import_name, name,
                         description, version, author, type, settings)
        
    def add_routes(self):
        @self.route('/bulk', methods=['POST'])
        @jwt_required()
        def create_inventory():
            current_user = get_jwt_identity()
            body = request.get_json()

            if 'post_type' not in body:
                return {'msg': 'No se especificó el tipo de contenido'}, 400

            if not self.has_role('admin', current_user) and not self.has_role('processing', current_user):
                return {'msg': 'No tiene permisos suficientes'}, 401

            task = self.bulk.delay(body, current_user)
            self.add_task_to_user(
                task.id, 'documentSegment.bulk', current_user, 'msg')

            return {'msg': 'Se agregó la tarea a la fila de procesamientos'}, 201
        
        @self.route('/anomgenerate', methods=['POST'])
        @jwt_required()
        def anomgenerate():
            current_user = get_jwt_identity()
            body = request.get_json()

            if 'id' not in body:
                return {'msg': 'No se especificó el archivo'}, 400

            if not self.has_role('admin', current_user) and not self.has_role('processing', current_user):
                return {'msg': 'No tiene permisos suficientes'}, 401
            
            task = self.anom.delay(body, current_user)
            self.add_task_to_user(task.id, 'documentSegment.anomgenerate', current_user, 'file_download')
            
            return {'msg': 'Se agregó la tarea a la fila de procesamientos'}, 201
        
        @self.route('/filedownload/<taskId>', methods=['GET'])
        @jwt_required()
        def file_download(taskId):
            current_user = get_jwt_identity()

            if not self.has_role('admin', current_user) and not self.has_role('processing', current_user):
                return {'msg': 'No tiene permisos suficientes'}, 401
            
            # Buscar la tarea en la base de datos
            task = mongodb.get_record('tasks', {'taskId': taskId})
            # Si la tarea no existe, retornar error
            if not task:
                return {'msg': 'Tarea no existe'}, 404
            
            if task['user'] != current_user and not self.has_role('admin', current_user):
                return {'msg': 'No tiene permisos para obtener la tarea'}, 401

            if task['status'] == 'pending':
                return {'msg': 'Tarea en proceso'}, 400

            if task['status'] == 'failed':
                return {'msg': 'Tarea fallida'}, 400

            if task['status'] == 'completed':
                if task['resultType'] != 'file_download':
                    return {'msg': 'Tarea no es de tipo file_download'}, 400
                
            path = USER_FILES_PATH + task['result']
            return send_file(path, as_attachment=True)
   

    @shared_task(ignore_result=False, name='documentSegment.bulk')
    def bulk(body, user):

        filters = {
            'post_type': body['post_type']
        }

        if 'parent' in body:
            if body['parent']:
                filters['parents.id'] = body['parent']

        # obtenemos los recursos
        resources = list(mongodb.get_all_records(
            'resources', filters, fields={'_id': 1}))
        resources = [str(resource['_id']) for resource in resources]

        records_filters = {
            'parent.id': {'$in': resources},
            'processing.fileProcessing': {'$exists': True},
            '$or': [{'processing.fileProcessing.type': 'document'}]
        }
        if not body['overwrite']:
            records_filters['processing.documentSegment'] = {'$exists': False}

        records = list(mongodb.get_all_records('records', records_filters, fields={
            '_id': 1, 'mime': 1, 'filepath': 1, 'processing': 1}))

        if len(records) > 0:
            model = lp.Detectron2LayoutModel(models_path + '/config_1.yaml',
                                             models_path + '/mymodel_1.pth',
                                             extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
                                             label_map={0: "Label", 1: "Manuscrito", 2: "Text"})
            
            ocr_agent = lp.TesseractAgent(languages='spa')

            for record in records:

                path = WEB_FILES_PATH + '/' + \
                    record['processing']['fileProcessing']['path'] + '/web/big'
                path_original = ORIGINAL_FILES_PATH + '/' + \
                    record['processing']['fileProcessing']['path'] + '.pdf'
                
                files = os.listdir(path)
                page = 0
                resp = []

                

                for f in files:
                    words = None
                    image = cv2.imread(path + '/' + f)
                    image = image[..., ::-1]
                    image_width = image.shape[1]
                    image_height = image.shape[0]
                    aspect_ratio = image_width / image_height

                    page += 1

                    layout = model.detect(image)

                    text_blocks = lp.Layout(
                        [b for b in layout if b.type == 'Text'])
                    manual_blocks = lp.Layout(
                        [b for b in layout if b.type == 'Manuscrito'])
                    label_blocks = lp.Layout(
                        [b for b in layout if b.type == 'Label'])
                    

                    resp_page = []

                    
                    
                    def get_obj(b, txt, type, segment_words):
                        obj = {
                            'text': txt,
                            'type': type,
                            'bbox': {
                                'x': b.block.x_1 / image_width,
                                'y': b.block.y_1 / image_height,
                                'width': (b.block.x_2 - b.block.x_1) / image_width,
                                'height': (b.block.y_2 - b.block.y_1) / image_height
                            },
                            'words': [{
                                'text': s['text'],
                                'bbox': {
                                    'x': s['left'] / image_width,
                                    'y': s['top'] / image_height,
                                    'width': s['width'] / image_width,
                                    'height': s['height'] / image_height
                                }
                            } for s in segment_words]
                        }
                        return obj
                    
                    def segment_image(b, image):
                        segment_image = (
                            b.pad(left=5, right=5, top=5,
                                bottom=5).crop_image(image)
                        )

                        segment_text = ocr_agent.detect(
                            segment_image, return_response=True, return_only_text=False)
                        
                        txt = segment_text['text']
                        words = []

                        for index, row in segment_text['data'].iterrows():
                            if row['text'] != '':
                                words.append({
                                    'text': row['text'],
                                    'left': row['left'],
                                    'top': row['top'],
                                    'width': row['width'],
                                    'height': row['height']
                                })

                        for w in words:
                            w['left'] += b.block.x_1
                            w['top'] += b.block.y_1

                        return txt, words
                    

                    

                    for b in text_blocks:
                        txt, segment_words = segment_image(b, image)

                        obj = get_obj(b, txt, 'text', segment_words)

                        resp_page.append(obj)
                    
                    for b in manual_blocks:
                        txt = ''
                        segment_words = []

                        obj = get_obj(b, txt, 'manuscrito', segment_words)

                        resp_page.append(obj)
                    
                    for b in label_blocks:
                        txt, segment_words = segment_image(b, image)

                        obj = get_obj(b, txt, 'label', segment_words)

                        resp_page.append(obj)

                    resp.append({
                        'page': page,
                        'blocks': resp_page
                    })

                update = {
                    'processing': record['processing']
                }

                update['processing']['documentSegment'] = {
                    'type': 'docseg_extraction',
                    'result': resp
                }

                update = RecordUpdate(**update)
                mongodb.update_record(
                    'records', {'_id': record['_id']}, update)

        update_cache_records()
        update_cache_resources()
        return 'Extracción de texto finalizada'

    @shared_task(ignore_result=False, name='documentSegment.anomgenerate')
    def anom(body, user):
        record = mongodb.get_record('records', {'_id': ObjectId(body['id'])}, fields={
            '_id': 1, 'mime': 1, 'filepath': 1, 'processing': 1
        })

        if 'processing' not in record:
            return {'msg': 'Registro no tiene procesamientos'}, 404
        if 'fileProcessing' not in record['processing']:
            return {'msg': 'Registro no tiene procesamiento de archivo'}, 404
        if 'documentSegment' not in record['processing']:
            return {'msg': 'Registro no tiene procesamiento de extracción de entidades nombradas'}, 404

        path = WEB_FILES_PATH + '/' + \
                    record['processing']['fileProcessing']['path'] + '/web/big'
        path_original = ORIGINAL_FILES_PATH + '/' + \
                    record['processing']['fileProcessing']['path'] + '.pdf'
        
        files = os.listdir(path)

        folder_path = USER_FILES_PATH + '/' + user + '/documentSegmentAnom'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_id = str(uuid.uuid4())

        pdf_path = folder_path + '/' + file_id + '.pdf'

        step = 0
        images = []

        pages = record['processing']['documentSegment']['result']

        hidden_labels = [b['id'] for b in body['type']]

        for file in files:
            step += 1
            page = pages[step - 1]
            img = Image.open(os.path.join(path, file))
            draw = ImageDraw.Draw(img)

            for block in page['blocks']:
                labels = [block['type']]
                if any(label in hidden_labels for label in labels):
                    # draw a rectangle around the block in the image, the coordinates are relative to the image and are x,y,width,height
                    x = block['bbox']['x'] * img.width
                    y = block['bbox']['y'] * img.height
                    w = block['bbox']['width'] * img.width
                    h = block['bbox']['height'] * img.height
                    draw.rectangle((x, y, x + w, y + h), fill='black')
                    # write the label in the image
                    # draw.text((x + (w / 2), y + (h / 2)), block['type'], fill='black')


            images.append(img)
        

        images[0].save(
            pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:]
        )



        return '/' + user + '/documentSegmentAnom/' + file_id + '.pdf'

plugin_info = {
    'name': 'Segmentación en documentos escaneados PDF',
    'description': 'Plugin para identificar segmentos de un documento y generar una versión anonimizada.',
    'version': '0.1',
    'author': 'Néstor Andrés Peña',
    'type': ['bulk'],
    'settings': {
        'settings_bulk': [
            {
                'type':  'instructions',
                'title': 'Instrucciones',
                'text': 'Este plugin permite identificar segmentos de un documento escaneado en formato PDF y generar una versión anonimizada del mismo. Para ello, se debe seleccionar el tipo de segmento a identificar y el documento a procesar. El resultado de la segmentación se puede descargar en formato PDF.',
            },
            {
                'type': 'checkbox',
                'label': 'Sobreescribir procesamientos existentes',
                'id': 'overwrite',
                'default': False,
                'required': False,
            }
        ],
        'settings_detail': [
            {
                'type': 'multicheckbox',
                'label': 'Entidades nombradas a extraer',
                'id': 'type',
                'default': [],
                'options': [
                    {
                        'label': 'Manuscrito',
                        'value': 'manuscrito'
                    }
                ]
            },
            {
                'type': 'button',
                'label': 'Generar versión anonimizada',
                'id': 'generate',
                'callback': 'anomgenerate'
            }
        ]
    }
}
