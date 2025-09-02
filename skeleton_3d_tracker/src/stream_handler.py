import numpy as np
import cv2
from is_wire.core import Subscription, Channel
from is_msgs.image_pb2 import ObjectAnnotations, Image
from streamChannel import StreamChannel # A classe que você já utiliza

class StreamHandler:
    """
    Gerencia a comunicação com o broker, incluindo a conexão, subscrição
    em tópicos, e o consumo e preparação inicial das mensagens.
    """
    def __init__(self, config: dict):
        """
        Estabelece a conexão com o broker e se inscreve nos tópicos de
        imagem e detecção de esqueletos para todas as câmeras.

        Args:
            config (dict): Dicionário de configuração contendo 'broker_uri' e 'num_cameras'.
        """
        self.config = config
        self.image_channels = []
        self.skeleton_channels = []
        self._subscriptions = [] # Armazena as subscrições para mantê-las ativas

        print(f"Conectando ao broker em {self.config['broker_uri']}...")
        for i in range(1, self.config['num_cameras'] + 1):
            # Configurar canal e subscrição para as imagens
            img_channel = StreamChannel(self.config['broker_uri'])
            self.image_channels.append(img_channel)
            img_sub = Subscription(channel=img_channel)
            img_sub.subscribe(topic=f'CameraGateway.{i}.Frame')
            self._subscriptions.append(img_sub)
            
            # Configurar canal e subscrição para os esqueletos
            sk_channel = StreamChannel(self.config['broker_uri'])
            self.skeleton_channels.append(sk_channel)
            sk_sub = Subscription(channel=sk_channel)
            sk_sub.subscribe(topic=f'SkeletonDetector.{i}.Detection')
            self._subscriptions.append(sk_sub)
        
        print(f"Subscrição realizada para {self.config['num_cameras']} câmeras.")

    def get_latest_messages(self):
        """
        Consome a última mensagem disponível de todos os canais.
        
        Returns:
            Um dicionário com as mensagens brutas {'images': [...], 'skeletons': [...]}
            ou None se alguma das mensagens ainda não chegou.
        """
        messages = {
            'images': [chan.consume_last() for chan in self.image_channels],
            'skeletons': [chan.consume_last() for chan in self.skeleton_channels]
        }
        
        # Verifica se todas as mensagens foram recebidas com sucesso
        all_messages = messages['images'] + messages['skeletons']
        if any(isinstance(msg, bool) or msg is None for msg in all_messages):
            return None # Retorna None se alguma mensagem estiver faltando
            
        return messages

    def _to_np(self, input_image: Image):
        """Converte uma mensagem de imagem para um array NumPy."""
        if isinstance(input_image, np.ndarray):
            return input_image
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

    def _calib_img_from_file(self, npzCalib: np.lib.npyio.NpzFile, image: np.ndarray) -> np.ndarray:
        """Aplica a calibração para remover a distorção da imagem."""
        undistort_img = cv2.undistort(image, npzCalib['K'], npzCalib['dist'], None, npzCalib['nK'])
        # Supondo que 'roi' está nos seus arquivos de calibração .npz
        if 'roi' in npzCalib:
            x, y, w, h = npzCalib['roi']
            return undistort_img[y:y+h, x:x+w]
        return undistort_img

    def prepare_input_data(self, messages: dict, calib_files: list):
        """
        Desempacota as mensagens brutas, converte imagens e corrige a distorção.
        
        Args:
            messages (dict): O dicionário de mensagens brutas de get_latest_messages.
            calib_files (list): Lista de arquivos de calibração .npz carregados.
        
        Returns:
            Uma tupla contendo (lista_de_imagens_corrigidas, lista_de_anotacoes_de_esqueletos).
        """
        # Desempacota as anotações dos esqueletos
        skeleton_annotations = [
            msg.unpack(ObjectAnnotations) for msg in messages['skeletons']
        ]

        # Converte, desempacota e corrige a distorção das imagens
        undistorted_images = []
        for i in range(self.config['num_cameras']):
            raw_image_msg = messages['images'][i]
            # Assumindo que a mensagem é do tipo 'is_msgs.Image' e precisa ser desempacotada
            unpacked_image = raw_image_msg.unpack(Image)
            
            # Converte para np.array e corrige a distorção
            np_image = self._to_np(unpacked_image)
            calibrated_image = self._calib_img_from_file(calib_files[i], np_image)
            undistorted_images.append(calibrated_image)

        return undistorted_images, skeleton_annotations