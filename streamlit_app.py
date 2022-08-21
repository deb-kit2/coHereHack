import streamlit as st
import cohere 
import pandas as pd
from sentence_transformers import util

class TextGenerator:
    def __init__(self):
        # check key
        self.co = cohere.Client('api')
        self.model = 'model'
        
        # self.prompt ='Simplify the below text:\nComplex input: However, lymphocytes, a type of white blood cell, will meet the antigens, or proteins, in the peripheral lymphoid organs, which includes lymph nodes. The antigens are displayed by specialized cells in the lymph nodes.\nSimplified output: Lymph nodes contain white blood cells.\n--separator--\nSimplify the below text:\nComplex input: Also known as knockout organisms or simply knockouts, they are used in learning about a gene that has been sequenced, but which has an unknown or incompletely known function.\nSimplified output: Knockout mice, are used to learn about a gene that has been sequenced, but whose function is unknown or incompletely known.\n--separator--  '
        self.prompt = "Simplify the below text:\n"

    def generate_text(self, starting_text: str, reference : str) -> str:
        self.prompt = self.prompt + starting_text

        # check keys
        response= self.co.generate(
            model = self.model,
            prompt = self.prompt,
            max_tokens = 50,
            temperature = 0.6,
            k = 500,
            p = 0,
            frequency_penalty = 0.33,
            presence_penalty = 0,
            stop_sequences = ["--separator--", "Simplified output:", "Complex output:", "Simple Comments:", "How to complete:", "Complex Input:"],
            return_likelihoods = 'GENERATION',
            num_generations = 5)

        gens = []
        for gen in response.generations:
            gens.append(gen.text)

        df = pd.DataFrame({'generation':gens})
        # # Drop duplicates
        df = df.drop_duplicates(subset=['generation'])

        t = []

        for _ in df["generation"] :
            t.append(_.split("--separator--")[0].strip())

        tembed = self.co.embed(model = 'small', texts = t)
        tembed = tembed.embeddings
        ref = self.co.embed(model = 'small', texts = [reference])
        ref = ref.embeddings[0]

        score = []
        for i, emb in enumerate(tembed) :
            similarity = float(util.pytorch_cos_sim(ref[ : len(emb)], emb))
            if similarity > 0.98 :
                continue

            score.append([similarity, i])

        score.sort(key = lambda y: y[0])

        return(t[score[-1][1]].split("Simplified")[0])


if __name__ == '__main__':
    st.title('Simplified universe')
    starting_text = st.text_area('Let us simiplify the world for you')
    generator = TextGenerator()

    
    if starting_text:
        reference = starting_text
        starting_text = "Simplify the below text:\nComplex input: " + starting_text + "\nSimplified output:"
        response = generator.generate_text(starting_text, reference)
        st.markdown(f'{response}')