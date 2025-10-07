# Deteccao-de-SPAM
Realização da Ponderada: Detecção de SPAM

# 1. Detecção de SPAM em Mensagens SMS com LSTM e GloVe

&emsp;Este projeto implementa um modelo de **rede neural recorrente (LSTM)** utilizando **vetores de palavras GloVe** para identificar mensagens de texto **SPAM**. Além disso, inclui um **modelo alternativo** baseado em **TF-IDF + Rede Neural Densa**, para comparação de desempenho.


# 2. Objetivos do Projeto

1. Carregar e explorar o dataset de mensagens (`spam.csv`);
2. Realizar **análise exploratória de dados** com gráficos;
3. Implementar um **pipeline de pré-processamento de texto** (limpeza, tokenização, lematização e remoção de stopwords);
4. Utilizar vetores **GloVe (Global Vectors for Word Representation)** para vetorização das mensagens;
5. Treinar um modelo **LSTM** para classificação binária (SPAM / HAM);
6. Avaliar o modelo usando **acurácia**, **relatório de classificação** e **matriz de confusão**;
7. Comparar os resultados com um modelo alternativo de **Rede Neural Densa** treinado sobre **TF-IDF**.


# 3. Dataset Utilizado

- **Nome:** SMS Spam Collection Dataset, dispoível no Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **Arquivo:** `spam.csv`  
- **Descrição:** Contém mais de 5.500 mensagens SMS rotuladas como `ham` (não-SPAM) ou `spam`.
- 

# 4. Pipeline de Pré-Processamento

1. **Normalização** de caracteres Unicode  
2. **Remoção** de URLs, e-mails, números e pontuação  
3. **Conversão para minúsculas**  
4. **Remoção de stopwords** (usando NLTK)  
5. **Lematização** (usando spaCy)  
6. Criação de novas colunas:
   - `text_clean` → texto limpo  
   - `text_final` → texto tokenizado e lematizado  


# 5.  Modelos Implementados

##  LSTM com GloVe
- Vetorização com embeddings **GloVe 100D**.
- Rede Neural Recorrente (LSTM) com dropout
- Treinamento supervisionado com `binary_crossentropy`

### Rede Neural Densa com TF-IDF
- Vetorização via **TF-IDF** (máx. 5000 features)
- Rede neural densa de 3 camadas com ReLU e dropout


## 5.1 Avaliação dos Modelos

&emsp;As métricas utilizadas foram:
- **Acurácia**
- **Matriz de Confusão**
- **Relatório de Classificação** (Precision, Recall e F1-Score)


# 6. Resultados

## 6.1 Modelo LSTM + GloVe

- **Acurácia:** 0.9749  
- **Matriz de confusão:**

|          | Predito Ham (0) | Predito Spam (1) |
|----------|-----------------|-----------------|
| **Ham (0)**  | 956             | 9               |
| **Spam (1)** | 19              | 131             |

- **Relatório de classificação:**

| Classe   | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Ham (0)  | 0.98      | 0.99   | 0.99     | 965     |
| Spam (1) | 0.94      | 0.87   | 0.90     | 150     |
| **Média geral** | 0.97      | 0.97   | 0.97     | 1115    |

**Interpretação:**  
O modelo LSTM com GloVe apresenta excelente desempenho geral.  
- A classe Ham é detectada com quase 99% de recall, mostrando poucos falsos negativos.  
- A classe Spam tem recall menor (87%), indicando que algumas mensagens de spam foram classificadas como ham.


## 6.2 Modelo TF-IDF + Rede Densa

- **Acurácia:** 0.9730  
- **Matriz de confusão:**

|          | Predito Ham (0) | Predito Spam (1) |
|----------|-----------------|-----------------|
| **Ham (0)**  | 958             | 7               |
| **Spam (1)** | 23              | 127             |

**Interpretação:**  
O modelo TF-IDF + Rede Densa também apresenta desempenho muito próximo ao LSTM, com acurácia similar.  
- O recall para Spam é ligeiramente menor (128 de 150 corretamente identificadas), mostrando um pouco mais de falsos negativos.  
- O modelo é ligeiramente melhor em detectar Ham (958 de 965).


## 6.3 Comparação entre os modelos

- A acurácia geral dos dois modelos é praticamente igual, **em torno de 97,4%**.  
- O **LSTM + GloVe** tem melhor recall para Spam, importante para minimizar mensagens de spam não detectadas.  
- O **TF-IDF + Rede Densa** é mais simples e computacionalmente mais leve, mas perde um pouco na detecção de Spam.

**Conclusão:**  
&emsp;Ambos os modelos são altamente eficientes para detecção de spam em mensagens SMS. A escolha pode depender de recursos computacionais e da prioridade em reduzir falsos negativos na classe Spam.


# 7. Próximos Passos

&emsp;Para aprimorar o desempenho e ampliar o escopo do projeto, recomenda-se:

-  **Explorar modelos baseados em Transformers**, como BERT ou DistilBERT, para comparação direta com LSTM.  
- **Aplicar técnicas de data augmentation** (sinônimos, tradução reversa, ruído textual) para aumentar a robustez do modelo.  
- **Implementar balanceamento de classes** (por exemplo, *SMOTE* ou *undersampling*) para evitar vieses caso o dataset seja desbalanceado.  
- **Desenvolver um pipeline de inferência em produção**, expondo o modelo como API (via FastAPI, Flask ou Gradio).  
- **Monitorar o desempenho em tempo real**, adicionando métricas de drift e logs para detectar degradação de performance.  
- **Treinar embeddings customizados** (Word2Vec ou FastText) com base em um corpus próprio de mensagens, avaliando ganho sobre o GloVe genérico.

# 8. Conclusão

&emsp;O experimento demonstrou a eficácia de modelos baseados em **redes neurais recorrentes** e **representações semânticas densas (GloVe)** para tarefas de **classificação de texto curta**, como SMS.

&emsp;Em síntese:
- A combinação de **embeddings pré-treinados + LSTM** proporcionou alta acurácia e boa generalização.
- Modelos mais simples (como TF-IDF + Densa) ainda oferecem **ótima relação custo-benefício**, sendo indicados para ambientes com recursos limitados.
- O pipeline proposto é **modular e reutilizável**, podendo ser facilmente adaptado para outros domínios de classificação textual (e-mails, comentários, redes sociais etc.).
