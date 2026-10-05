import unittest
import pandas as pd
import numpy as np
from datetime import datetime


from ecommerce_cleaner import (
    limpa_valor_compra,
    padroniza_datas,
    cria_feature_velocidade,
    filtra_idade_comprador,
    padroniza_navegadores,
    remove_transacoes_fantasmas
)

class TestFraudDataProcessing(unittest.TestCase):

    def setUp(self):
        
        self.raw_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5, 6],
            'signup_time': [
                '2026/01/14 10:00:00',
                '2026/01/15 12:00:00',
                '2026-01-16 10:00:00',
                '14-01-2026 10:00:00',
                '2026/01/18 10:00:00',
                '2026/01/19 10:00:00'
            ],
            'purchase_time': [
                '2026/01/14 10:30:00', # 30 min depois
                '2026/01/15 11:30:00', # -60 min 
                '2026-01-16 12:00:00',
                '14-01-2026 11:00:00',
                '2026/01/18 10:15:00',
                '2026/01/19 10:00:00'
            ],
            'purchase_value': ["150", "US$ 45.50", "50,00", "  100.00 ", 75.5, np.nan],
            'device_id': ['DEV_101', '', None, 'DEV_104', 'DEV_105', 'DEV_106'],
            'browser': ['Chrome', 'CHROME ', 'InternetExplorer', 'IE', 'Firefox', 'Opera'],
            'age': [25, 12, -5, 105, 30, 70]
        })

    # Desafio 1: Limpeza de purchase_value
    def test_limpa_valor_compra(self):
        v1 = limpa_valor_compra("150")
        v2 = limpa_valor_compra("US$ 45.50")
        v3 = limpa_valor_compra("50,00")
        
        self.assertAlmostEqual(v1, 150.0, places=2)
        self.assertAlmostEqual(v2, 45.50, places=2)
        self.assertAlmostEqual(v3, 50.00, places=2)
        self.assertIsInstance(v1, float)

    
    def test_padroniza_datas(self):
        df_result = padroniza_datas(self.raw_data.copy())
        
        self.assertEqual(df_result['signup_time'].dtype, np.dtype('datetime64[ns]'))
        self.assertEqual(df_result['purchase_time'].dtype, np.dtype('datetime64[ns]'))

   
    def test_cria_feature_velocidade(self):
        df_preparado = padroniza_datas(self.raw_data.copy())
        
        
        with self.assertRaises(ValueError):
            cria_feature_velocidade(df_preparado)
            
       
        df_valido = df_preparado.iloc[[0]].copy()
        df_res = cria_feature_velocidade(df_valido)
        
        self.assertIn('velocidade_compra', df_res.columns)
        self.assertTrue((df_res['velocidade_compra'] >= 0).all())
        self.assertAlmostEqual(df_res['velocidade_compra'].iloc[0], 30.0, places=2)

    
    def test_filtra_idade_comprador(self):
        df_res = filtra_idade_comprador(self.raw_data.copy())
        
       
        self.assertTrue(pd.isna(df_res.loc[1, 'age']))
        self.assertTrue(pd.isna(df_res.loc[2, 'age']))
        self.assertTrue(pd.isna(df_res.loc[3, 'age']))
        
        self.assertEqual(df_res.loc[0, 'age'], 25)
        self.assertEqual(df_res.loc[4, 'age'], 30)

    def test_padroniza_navegadores(self):
        df_res = padroniza_navegadores(self.raw_data.copy())
        browsers_unicos = set(df_res['browser'].unique())
        
      
        self.assertIn("Chrome", browsers_unicos)
        self.assertIn("IE", browsers_unicos)
        self.assertNotIn("CHROME ", browsers_unicos)
        self.assertNotIn("InternetExplorer", browsers_unicos)
        self.assertEqual(df_res.loc[1, 'browser'], "Chrome")
        self.assertEqual(df_res.loc[2, 'browser'], "IE")

  
    def test_remove_transacoes_fantasmas(self):
        df_res = remove_transacoes_fantasmas(self.raw_data.copy())
        
        
        linhas_iniciais = len(self.raw_data)
        linhas_invalidas_esperadas = 2
        
        self.assertEqual(len(df_res), linhas_iniciais - linhas_invalidas_esperadas)

if __name__ == '__main__':
    unittest.main()