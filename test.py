import unittest
from aquabot import *

COLETAR_AMOSTRA = "audios/coletar-amostra.wav"
INICIAR_EXPLORACAO = "audios/iniciar-exploracao.wav"
LANCAR_DRONE_SUBMARINO = "audios/lancar-drone-submarino.wav"
MAPEAR_FUNDO_OCEANICO = "audios/mapear-fundo-oceanico.wav"
RETORNAR_PARA_SUPERFICIE = "audios/retornar-para-superficie.wav"

class AquabotTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        started, cls.processor, cls.model, cls.recorder, cls.stop_words, cls.actions = start(cls.device)

    def test_coletar_amostra(self):
        transcription = transcribe(self.device, load_speech(COLETAR_AMOSTRA), self.model, self.processor)
        self.assertIsNotNone(transcription)

        command = remove_stop_words(transcription, self.stop_words)
        valid, action, object = validate_command(command, self.actions)

        self.assertTrue(valid)
        self.assertEqual(action, "coletar")
        self.assertEqual(object, "amostra")

    def test_iniciar_exploracao(self):
        transcription = transcribe(self.device, load_speech(INICIAR_EXPLORACAO), self.model, self.processor)
        self.assertIsNotNone(transcription)

        command = remove_stop_words(transcription, self.stop_words)
        valid, action, object = validate_command(command, self.actions)

        self.assertTrue(valid)
        self.assertEqual(action, "iniciar")
        self.assertEqual(object, "exploração")

    def test_lancar_drone_submarino(self):
        transcription = transcribe(self.device, load_speech(LANCAR_DRONE_SUBMARINO), self.model, self.processor)
        self.assertIsNotNone(transcription)

        command = remove_stop_words(transcription, self.stop_words)
        valid, action, object = validate_command(command, self.actions)

        self.assertTrue(valid)
        self.assertEqual(action, "lançar")
        self.assertEqual(object, "drone submarino")

    def test_mapear_fundo_oceanico(self):
        transcription = transcribe(self.device, load_speech(MAPEAR_FUNDO_OCEANICO), self.model, self.processor)
        self.assertIsNotNone(transcription)

        command = remove_stop_words(transcription, self.stop_words)
        valid, action, object = validate_command(command, self.actions)

        self.assertTrue(valid)
        self.assertEqual(action, "mapear")
        self.assertEqual(object, "fundo oceânico")

    def test_retornar_para_superficie(self):
        transcription = transcribe(self.device, load_speech(RETORNAR_PARA_SUPERFICIE), self.model, self.processor)
        self.assertIsNotNone(transcription)

        command = remove_stop_words(transcription, self.stop_words)
        valid, action, object = validate_command(command, self.actions)

        self.assertTrue(valid)
        self.assertEqual(action, "retornar")
        self.assertEqual(object, "superfície")


if __name__ == "__main__":
    unittest.main()
