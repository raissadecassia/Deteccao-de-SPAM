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

## 6.1 Comparação Geral de Acurácia

- **Acurácia LSTM + GloVe:** 0.9740  
- **Acurácia TF-IDF + Rede Densa:** 0.9722  

**Conclusão inicial:**  
&emsp;O modelo LSTM com GloVe apresentou melhor desempenho geral, com ligeira vantagem em relação ao modelo TF-IDF + Rede Densa.


## 6.2 Modelo LSTM + GloVe

- **Acurácia:** 0.9740  

- **Matriz de confusão:**

|          | Predito Ham (0) | Predito Spam (1) |
|----------|-----------------|-----------------|
| **Ham (0)**  | 959             | 6               |
| **Spam (1)** | 23              | 127            |


- **Relatório de classificação:**

| Classe   | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Ham (0)  | 0.98      | 0.99   | 0.99     | 965     |
| Spam (1) | 0.95      | 0.85   | 0.90     | 150     |
| **Média geral** | 0.97      | 0.97   | 0.97     | 1115    |

**Interpretação:**  
- O modelo detecta a classe Ham com altíssima precisão e recall.  
- Para a classe Spam, o recall é 85%, mostrando que algumas mensagens de spam foram classificadas como Ham.  
- O F1-score de Spam (0.90) mostra que o modelo ainda tem bom equilíbrio entre precisão e recall.


### 6.3 Modelo TF-IDF + Rede Densa

- **Acurácia:** 0.9722  

- **Matriz de confusão:**

|          | Predito Ham (0) | Predito Spam (1) |
|----------|-----------------|-----------------|
| **Ham (0)**  | 956             | 9               |
| **Spam (1)** | 22              | 128             |


- **Relatório de classificação:**

| Classe   | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Ham (0)  | 0.98      | 0.99   | 0.98     | 965     |
| Spam (1) | 0.93      | 0.85   | 0.89     | 150     |
| **Média geral** | 0.97      | 0.97   | 0.97     | 1115    |

**Interpretação:**  
- O modelo TF-IDF + Rede Densa tem desempenho muito próximo ao LSTM + GloVe.  
- O recall para Spam é igual (85%), mas a precisão é ligeiramente menor (0.93), refletindo mais falsos positivos.  
- É uma alternativa computacionalmente mais leve, porém com pequena perda na detecção de Spam.

### Conclusão da Comparação

- Ambos os modelos alcançam alta acurácia (~97%).  
- **LSTM + GloVe** se destaca por ligeira vantagem em precisão da classe Spam.  
- **TF-IDF + Rede Densa** é viável e eficiente, mas perde um pouco em sensibilidade para spam.  
- A escolha entre os modelos pode depender de **recursos computacionais** e **prioridade na detecção de Spam**.


# 7. Próximos Passos

&emsp;Para aprimorar o desempenho e ampliar o escopo do projeto, recomenda-se:

-  **Explorar modelos baseados em Transformers**, como BERT ou DistilBERT, para comparação direta com LSTM.  
- **Aplicar técnicas de data augmentation** (sinônimos, tradução reversa, ruído textual) para aumentar a robustez do modelo.  
- **Implementar balanceamento de classes** (por exemplo, *SMOTE* ou *undersampling*) para evitar vieses caso o dataset seja desbalanceado.  
- **Desenvolver um pipeline de inferência em produção**, expondo o modelo como API (via FastAPI, Flask ou Gradio).  
- **Monitorar o desempenho em tempo real**, adicionando métricas de drift e logs para detectar degradação de performance.  
- **Treinar embeddings customizados** (Word2Vec ou FastText) com base em um corpus próprio de mensagens, avaliando ganho sobre o GloVe genérico.

# 8. Conclusão

&emsp;O experimento demonstrou a eficácia dos modelos para tarefas de **classificação de texto curta**, como SMS.

&emsp;Em síntese:
- A combinação de **embeddings pré-treinados + LSTM** proporcionou alta acurácia e boa generalização.
- Modelos mais simples (como TF-IDF + Densa) ainda oferecem **ótima relação custo-benefício**, sendo indicados para ambientes com recursos limitados.
- O pipeline proposto é **modular e reutilizável**, podendo ser facilmente adaptado para outros domínios de classificação textual (e-mails, comentários, redes sociais etc.).
