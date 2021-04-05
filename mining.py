import nltk

base = [('acho que você deveria ser mais alegre','Gaslighting'),
		('eu tenho o direito de agir assim','Gaslighting'),
		('o problema é que você é emotivo','Gaslighting'),
		('você deveria aprender a brincar','Gaslighting'),
		('você é muito sensível','Gaslighting'),
		('você não acha que deveria ser menos exagerado','Gaslighting'),
		('você está sendo louco','Gaslighting'),
		('você precisa parar de ser implicante','Gaslighting'),
		('você precisa ser mais tranquilo','Gaslighting'),
		('desculpa eu não pensei antes de fazer isso','Argumento'),
		('eu deveria ter considerado os seus sentimentos','Argumento'),
		('eu não agirei assim novamente','Argumento'),
		('eu não considerei como você iria se sentir','Argumento'),
		('lamento isso não irá se repetir','Argumento'),
		('me desculpe não foi a minha intenção','Argumento'),
		('me perdoe por ter agido dessa forma','Argumento'),
		('sinto muito eu agi por impulso','Argumento'),
		('sinto muito eu não sabia que isso iria te chatear','Argumento')]

stopwords = ['a','acha','acho','agi','agirei', 'antes', 'assim','como','de','dessa','é','está', 'fazer',
			 'foi','forma','irá','iria','isso','me','minha','o','os','por','que','se','ser','te','tenho','ter']

# aplica stemmer desconsiderando as stopwords
def aplicastemmer(base):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (frase, classificacao) in base:
        comstemming = [str(stemmer.stem(p)) for p in frase.split() if p not in stopwords]
        frasesstemming.append((comstemming, classificacao))
    return frasesstemming

# guarda o retorno do método para a aplicação de steammer na base desconsiderando as stopwords
comstemming = aplicastemmer(base)

# lista de frases com stemming, sem stopwords + suas respectivas classificações
# print(comstemming)

# retorna todas as palavras da base
def buscapalavras(frases):
    todaspalavras = []
    for (palavras, classificacao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

# guarda o retorno do método de busca por todas as palavras da base
palavras = buscapalavras(comstemming)
# print(palavras)

# retorna a frequência com que cada palavra aparece
def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

# guarda o retorno do método de busca pela frequência de todas as palavras da base
frequencia = buscafrequencia(palavras)
# print(frequencia.most_common(50))

# retorna todas as palavras da base sem repetições
def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

# guarda o retorno do método de busca pela frequência de todas as palavras da base
palavrasunicas = buscapalavrasunicas(frequencia)
# print(palavrasunicas)

# retorna a ocorrência\não ocorrência de cada palavra da base no documento de entrada
def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicas:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

caracteristicasfrase = extratorpalavras(['voc', 'dev', 'problem'])
print(caracteristicasfrase)

# guarda uma lista com a ocorrência\não ocorrência de todas as palavras da base para cada uma das frases
basecompleta = nltk.classify.apply_features(extratorpalavras, comstemming)
# print(basecompleta[1])

# constrói a tabela de probabilidades da base utilizando o classificador Naïve Bayes
classificador = nltk.NaiveBayesClassifier.train(basecompleta)

# print(classificador.labels())
# print(classificador.show_most_informative_features())

# frase de entrada
teste = 'Você está exagerando'
print('Frase original: ' + teste)

def aplicastemmerfrase(frase):
    stemmer = nltk.stem.RSLPStemmer()
    frasestemming = []
    for (palavras) in frase.split():
        comstem = [p for p in palavras.split()]
        frasestemming.append(str(stemmer.stem(comstem[0])))
    return frasestemming

# print(teste)
print('Frase com stemming: ' + str(aplicastemmerfrase(teste)))

# novo = extratorpalavras(aplicastemmerfrase(teste))
# print(novo)

# retorna label
print('Essa frase foi classificada como: ' + classificador.classify(extratorpalavras(aplicastemmerfrase(teste))))

# retorna labels e probabilidades
# distribuicao = classificador.prob_classify(extratorpalavras(aplicastemmerfrase(teste)))
for classe in classificador.prob_classify(extratorpalavras(aplicastemmerfrase(teste))).samples():
    print("%s: %f" % (classe, classificador.prob_classify(extratorpalavras(aplicastemmerfrase(teste))).prob(classe)))