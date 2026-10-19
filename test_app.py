"""
=============================================================================
SUITE DE PRUEBAS UNITARIAS Y VERIFICACIÓN
Aplicación: app.py (Prosegur Predictor de Notas de Crédito)
=============================================================================
"""

import os
import unittest
import py_compile
import numpy as np
import pandas as pd
from PIL import Image

# Parche de compatibilidad antes de importar
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

import joblib


class TestProsegurApp(unittest.TestCase):

    def setUp(self):
        self.model_path = os.path.join("modelo_nc", "modelo_nc_mes_siguiente_v2.joblib")
        self.logo_png = os.path.join("recursos_visuales", "logo.png")
        self.logo_jpg = os.path.join("recursos_visuales", "logo_texto.jpg")
        self.template_csv = "template_evaluacion_masiva.csv"
        self.app_path = "app.py"

    def test_01_assets_exist(self):
        """Verifica que los logos e imágenes requeridas existan y sean válidas."""
        self.assertTrue(os.path.exists(self.logo_png), "Falta el archivo recursos_visuales/logo.png")
        self.assertTrue(os.path.exists(self.logo_jpg), "Falta el archivo recursos_visuales/logo_texto.jpg")
        
        # Validar apertura con PIL
        with Image.open(self.logo_png) as img:
            self.assertEqual(img.format, "PNG")
            self.assertGreater(img.size[0], 0)
        with Image.open(self.logo_jpg) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertGreater(img.size[0], 0)

    def test_02_model_package_integrity(self):
        """Verifica que el archivo joblib contenga todas las claves y tipos esperados."""
        self.assertTrue(os.path.exists(self.model_path), f"No existe el archivo {self.model_path}")
        pkg = joblib.load(self.model_path)
        self.assertIsInstance(pkg, dict, "El paquete serializado debe ser un diccionario.")

        required_keys = ['modelo', 'calibrador', 'umbral', 'features', 'num', 'cat']
        for k in required_keys:
            self.assertIn(k, pkg, f"Falta la clave requerida '{k}' en el paquete del modelo.")

        self.assertIsInstance(pkg['features'], list)
        self.assertEqual(len(pkg['features']), 27, "El modelo debe requerir exactamente 27 features.")
        self.assertIsInstance(pkg['umbral'], (float, int, np.floating))
        self.assertGreater(pkg['umbral'], 0.0)
        self.assertLess(pkg['umbral'], 1.0)

    def test_03_single_prediction_logic(self):
        """Prueba una predicción individual con datos sintéticos y verifica rango de probabilidad."""
        from app import predict_dataframe, load_model
        
        pkg, err = load_model(self.model_path)
        self.assertIsNone(err, f"Error cargando modelo: {err}")
        self.assertIsNotNone(pkg)

        # Crear fila sintética de prueba
        sample_row = {
            'cliente': 'CLI-UNIT-001',
            'n_nc_1m': 1, 'n_nc_3m': 2, 'n_nc_6m': 3, 'n_nc_hist': 10,
            'monto_1m': 500.0, 'monto_3m': 1200.0, 'monto_6m': 2000.0, 'monto_hist': 6000.0,
            'meses_con_nc_6m': 2, 'meses_desde_ultima': 1.0, 'antiguedad_meses': 12.0,
            'monto_prom_nota': 600.0, 'nc_lag1': 1, 'nc_lag2': 0, 'nc_lag3': 0, 'nc_lag5': 1, 'nc_lag11': 0,
            'n_nc_12m': 4, 'frac_meses_con_nc': 0.25, 'tendencia_3v3': 1,
            'delegacion': 'DELEGACION LIMA', 'clasificacion': 'NORMAL',
            'transaccion': 'Recurrente', 'tipo_servicio': 'ALARMAS',
            'motivo_frec': 'DESCUENTO POR ITEM', 'area_frec': 'ATC', 'mes_objetivo': '11'
        }
        df_in = pd.DataFrame([sample_row])
        df_out = predict_dataframe(df_in, pkg)

        self.assertIn('prob_nc_mes_siguiente', df_out.columns)
        self.assertIn('alerta', df_out.columns)
        self.assertIn('nivel_riesgo', df_out.columns)

        prob = df_out.iloc[0]['prob_nc_mes_siguiente']
        alerta = df_out.iloc[0]['alerta']
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertIn(alerta, [0, 1])
        self.assertEqual(alerta, int(prob >= pkg['umbral']))

    def test_04_batch_prediction_template(self):
        """Verifica que la plantilla CSV masiva cargue y procese adecuadamente."""
        from app import predict_dataframe, load_model

        pkg, _ = load_model(self.model_path)
        self.assertTrue(os.path.exists(self.template_csv), "Falta template_evaluacion_masiva.csv")
        
        df_batch = pd.read_csv(self.template_csv)
        self.assertGreater(len(df_batch), 0, "La plantilla debe contener al menos 1 fila de ejemplo.")
        
        df_out = predict_dataframe(df_batch, pkg)
        self.assertEqual(len(df_out), len(df_batch))
        for p in df_out['prob_nc_mes_siguiente']:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_05_graceful_missing_and_unknown_categories(self):
        """Verifica la robustez ante columnas ausentes o categorías nunca antes vistas."""
        from app import predict_dataframe, load_model

        pkg, _ = load_model(self.model_path)
        # DataFrame con solo 2 columnas y categorías desconocidas
        df_sparse = pd.DataFrame([
            {'cliente': 'CLI-DESCONOCIDO', 'delegacion': 'DELEGACION INEXISTENTE_MARTE', 'monto_1m': 9999.0}
        ])
        
        # No debe lanzar excepción
        df_out = predict_dataframe(df_sparse, pkg)
        self.assertEqual(len(df_out), 1)
        prob = df_out.iloc[0]['prob_nc_mes_siguiente']
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_06_app_syntax_and_compilation(self):
        """Verifica que app.py compile sin errores de sintaxis."""
        try:
            py_compile.compile(self.app_path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"Fallo de compilación en {self.app_path}: {e}")

    def test_07_preset_state_flow(self):
        """Verifica que los botones de presets actualicen el session_state y persistan al ejecutar."""
        from streamlit.testing.v1 import AppTest
        
        at = AppTest.from_file(self.app_path)
        at.run()

        # El estado inicial debe ser promedio
        self.assertEqual(at.session_state.get("inp_cod_cliente"), "CLI-PROM-105")

        # Simular clic en botón Alto Riesgo (primer botón de los presets)
        # Los primeros 3 botones son los presets
        btn_alto = [b for b in at.button if "Alto Riesgo" in b.label][0]
        btn_alto.click().run()
        self.assertEqual(at.session_state.get("inp_cod_cliente"), "CLI-ALTO-999")
        self.assertEqual(at.session_state.get("preset_nombre"), "alto")

        # Simular clic en submit del formulario para comprobar persistencia
        form_submit = at.form_submit_button[0]
        form_submit.click().run()
        # Debe mantenerse el cliente de alto riesgo y no volver a promedio
        self.assertEqual(at.session_state.get("inp_cod_cliente"), "CLI-ALTO-999")

        # Simular clic en botón Saludable
        btn_salud = [b for b in at.button if "Saludable" in b.label][0]
        btn_salud.click().run()
        self.assertEqual(at.session_state.get("inp_cod_cliente"), "CLI-SALUD-001")


if __name__ == '__main__':
    unittest.main()

