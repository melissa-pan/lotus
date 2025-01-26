
import os
import time
import ast

import numpy as np
import pandas as pd
from biodex_prompts import map_dem_df
from datasets import load_dataset
from metrics import compute_precision, compute_rank_precision, compute_recall
from pipeline_tester import Pipeline, PipelineTester
from scipy.linalg import lstsq

import lotus
from lotus.models import LM, SentenceTransformersRM, LiteLLMRM
from lotus.types import CascadeArgs


def learn_linear_transformation(S, A):
    # copy S and A
    S = S.copy()
    A = A.copy()
    # Flatten matrices S and A
    S_flat = S.reshape(-1, S.shape[-1])  # Reshape S into a 2D array
    A_flat = A.reshape(-1, A.shape[-1])  # Reshape A into a 2D array

    # Use least squares regression to find the linear transformation
    transformation, _, _, _ = lstsq(S_flat, A_flat)

    return transformation.reshape(S.shape[-1], A.shape[-1])  # Reshape the result into a matrix


class BiodexTester(PipelineTester):
    def __init__(self, n_train_samples=11543, n_samples=4249, truncation_limit=8000):
        self.truncation_limit = truncation_limit
        print("Using truncation limit of: ", self.truncation_limit)
        self.train_queries = self.load_train_queries(n_train_samples)
        return super().__init__(n_samples)
        # intializes self.queries_df, self.corpus_df, self.results_dir, self.pipelines

    def set_results_dir(self):
        return "biodex_results"

    def set_configs(self):
        rm = SentenceTransformersRM(model="intfloat/e5-base-v2", max_batch_size=4)
        # rm = LiteLLMRM(model="text-embedding-3-small")

        lm = LM(
            model="hosted_vllm/meta-llama/Meta-Llama-3-70B-Instruct",
            api_base="http://localhost:8200/v1/",
            max_batch_size=64,
            temperature=0.0,
            max_tokens=256,
        )
        # lm = LM(model="gpt-4o-mini-2024-07-18",
        #         max_batch_size=64,
        #         temperature=0.0,
        #         max_tokens=256,)

        lotus.settings.configure(
            lm=lm,
            rm=rm,
        )
        
        print(f"lotus.settings.lm.max_batch_size = {lotus.settings.lm.max_batch_size}")
        print(f"lotus.settings.lm.max_tokens = {lotus.settings.lm.max_tokens}")
        print(f"lotus.settings.lm.temperature = {lotus.settings.lm.kwargs['temperature']}")

    def load_train_queries(self, n_samples):
        df = load_dataset("BioDEX/BioDEX-Reactions", split="train").to_pandas()

        # split and remove trailing or leading whitespace
        df["reactions_list"] = df["reactions"].apply(lambda x: x.split(","))
        df["reactions_list"] = df["reactions_list"].apply(lambda x: [r.strip() for r in x])
        df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))
        df = df.load_sem_index("abstract", "biodex_abstract_e5")

        # truncate the fulltext to 8000 chars
        df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[: self.truncation_limit])

        # print the max number of characters in the truncated fulltext
        print(df["patient_description"].apply(len).max())

        return df[:n_samples]

    def load_queries(self, n_samples):
        df = load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()

        # split and remove trailing or leading whitespace
        df["reactions_list"] = df["reactions"].apply(lambda x: x.split(","))
        df["reactions_list"] = df["reactions_list"].apply(lambda x: [r.strip() for r in x])
        df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))
        df = df.load_sem_index("abstract", "biodex_abstract_e5")

        # truncate the fulltext to 8000 chars
        df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[: self.truncation_limit])

        # print the max number of characters in the truncated fulltext
        print(df["patient_description"].apply(len).max())

        return df[:n_samples]

    def load_map_df(self, filename="map_df.csv"):
        df = pd.read_csv(filename)

        # split and remove trailing or leading whitespace
        df["reactions_list"] = df["reactions"].apply(lambda x: x.split(","))
        df["reactions_list"] = df["reactions_list"].apply(lambda x: [r.strip() for r in x])
        df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))
        df = df.load_sem_index("abstract", "biodex_abstract_e5")

        # truncate the fulltext to 8000 chars
        df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[: self.truncation_limit])

        # print the max number of characters in the truncated fulltext
        print(df["patient_description"].apply(len).max())

        return df

    def load_corpus(self):
        reactions_df = pd.read_csv("biodex-reactions.csv")
        reactions_df.load_sem_index("reaction", "biodex_reactions_e5")
        return reactions_df

    # TODO this should be refactored to take a qa_df
    def compute_metrics(self, res_df, gt_col_name="reactions_list", pred_col_name="pred_reaction") -> pd.DataFrame:
        res_df["rank_precision@5"] = res_df.apply(
            lambda x: compute_rank_precision(x[gt_col_name], x[pred_col_name], cutoff=5),
            axis=1,
        )
        res_df["rank-precision@10"] = res_df.apply(
            lambda x: compute_rank_precision(x[gt_col_name], x[pred_col_name], cutoff=10),
            axis=1,
        )
        res_df["rank-precision@25"] = res_df.apply(
            lambda x: compute_rank_precision(x[gt_col_name], x[pred_col_name], cutoff=25),
            axis=1,
        )

        for k in [5, 10, 20, 50, 100, 200, 300, 400, 500]:
            res_df[f"recall@{k}"] = res_df.apply(lambda x: compute_recall(x[gt_col_name], x[pred_col_name], k), axis=1)

        res_df["precision@5"] = res_df.apply(lambda x: compute_precision(x[gt_col_name], x[pred_col_name], 5), axis=1)
        res_df["precision@10"] = res_df.apply(lambda x: compute_precision(x[gt_col_name], x[pred_col_name], 10), axis=1)

        res_df["num_ids"] = res_df.apply(lambda x: len(x[pred_col_name]), axis=1)

        # take subset of df with metrics
        df = res_df[[col for col in res_df.columns if "@" in col or "latency" in col or "num_ids" in col]]

        return df

    def get_map_df(self):
        map_df = self.queries_df.sem_map(
            "given the {abstract} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
            suffix="map_preds",
            examples=map_dem_df,
        )
        return map_df



class JoinCascade(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, range=250, name="llama"):
        # lotus.settings.
        start_t = time.time()

        # "given the {abstract} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
        map_instruction = "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions."
        join_instruction = "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?"
        print(f"corpus_df = {corpus_df.shape}")

        cascade_args = CascadeArgs(
            recall_target=recall_target,
            precision_target=precision_target,
            sampling_percentage=0.0001,
            map_instruction=map_instruction, map_examples=map_dem_df,
            cascade_IS_weight=0.9,
            cascade_IS_random_seed=42,
            cascade_IS_max_sample_range=range
        )
        answers_df, stats = queries_df.sem_join(corpus_df, join_instruction, cascade_args=cascade_args, return_stats=True)

        end_t = time.time()

        answers_df.to_csv(f"biodex_cascade_answers_for_lm_rerank_{name}.csv", index=True)
        
        print(f"Time taken: {end_t - start_t}")
        print(f"answers_df = {answers_df.columns.tolist()}")
        print(f"stats = {stats}")

        # post process for checking answers
        # convert reaction_list to string so we can groupby
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(
            lambda x: ", ".join(x)
        )
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list"])
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # convert reaction list from string back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(
            lambda x: x.split(", ")
        )

        return grouped_df, (end_t - start_t)  # qid


class rerankJoinCascade(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, range=250, name="Llama", top_k=25):
        start_t = time.time()
        
        # "given the {abstract} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
        map_instruction = "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions."
        join_instruction = "given the {patient_description} of a medical article, is the patient likely adversely affected by the {reaction}?"
        print(f"corpus_df = {corpus_df.shape}")
        
        cascade_args = CascadeArgs(
            recall_target=recall_target, 
            precision_target=precision_target,
            sampling_percentage=0.0001,
            map_instruction=map_instruction, map_examples=map_dem_df,
            cascade_IS_weight=0.9,
            cascade_IS_random_seed=42,
            cascade_IS_max_sample_range=range
        )
        answers_df, stats = queries_df.sem_join(corpus_df, join_instruction, cascade_args=cascade_args, return_stats=True)
        
        end_t_1 = time.time()
        
        # Rerank the answer with LLM
        def to_comma_separated(val):
            """
            Safely convert val (which could be a list or a string representing a list)
            into a comma-separated string.
            """
            
            if isinstance(val, list):
                # Already a list of strings
                return ", ".join(val)
            elif isinstance(val, str):
                # Possibly a string like "['foo', 'bar']"
                # Try literal_eval to see if it's a valid Python list
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        # It's a real list, convert to comma-separated
                        return ", ".join(parsed)
                    else:
                        # Not a list—just return the original string
                        return val
                except (SyntaxError, ValueError):
                    # It's not a parseable list, return as is
                    return val
            else:
                # Fallback: convert whatever it is to string
                return str(val)

        # 2) Normalize reactions_list so every row is a comma-separated string
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(to_comma_separated)

        # 3) Group by and aggregate
        grouped_df = (
            answers_df
            .groupby(["title", "abstract", "reactions", "reactions_list", "patient_description"], dropna=False)
            .apply(lambda grp: grp["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )

        # 4) Convert that comma-separated string (in grouped_df) back to a list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(
            lambda s: s.split(", ")
        )
        
        rerank_prompt = (
            f"Given {{patient_description}}, pick the {top_k} most applicable adverse drug reactions from the options "
            f"that are directly expressed in the following list: {{pred_reaction}}. "
            "Rank from most applicable to least applicable. "
            "Always write your answer as a list of comma-separated adverse drug reactions only and nothing else."
        )

        rerank_num_lm = grouped_df.shape[0]

        grouped_df = grouped_df.sem_map(
            rerank_prompt
        )
        
        end_t_2 = time.time()
        
        print(f"stats = {stats}")
        print(f"rerank used {rerank_num_lm} samples")
        print(f"Total Time taken: {end_t_2 - start_t}")
        print(f"no rerank took: {end_t_1 - start_t}")

        # Parse output
        known_prefixes = [
            f"Based on the patient description, the {top_k} most applicable adverse drug reactions are:\n\n",
            f"Based on the Patient_description, the {top_k} most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            f"Based on the Patient_description, the {top_k} most applicable adverse drug reactions are:\n\n",
            f"Based on the provided Patient_description, the {top_k} most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions:\n\n",
            "Here is the answer:\n\n",
            f"Here is the list of the {top_k} most applicable adverse drug reactions:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions from the options, ranked from most applicable to least applicable:"
        ]
        def remove_known_prefixes(text: str, prefixes: list) -> str:
            """
            Removes the first matching prefix from 'text' if found in 'prefixes',
            otherwise returns text unchanged.
            """
            for prefix in prefixes:
                if text.startswith(prefix):
                    return text[len(prefix):]
            return text
        
        grouped_df["_map"] = grouped_df["_map"].fillna("").apply(
            lambda x: remove_known_prefixes(x, known_prefixes)
        )
        grouped_df.rename(columns={"pred_reaction": "pred_reaction_norank"}, inplace=True)
        grouped_df["pred_reaction"] = grouped_df["_map"].apply(
            lambda x: [reaction.strip() for reaction in x.split(",") if reaction.strip()]
        )
        
        return grouped_df, (end_t_2 - start_t) 
    

class rerank(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, top_k=25, name=""):
        if name != "":
            answers_df = pd.read_csv(f"biodex_cascade_answers_for_lm_rerank_{name}.csv", index_col=0)
        else:
            answers_df = pd.read_csv("biodex_cascade_answers_for_lm_rerank.csv", index_col=0)

        start_t = time.time()

        # Rerank the answer with LLM
        def to_comma_separated(val):
            """
            Safely convert val (which could be a list or a string representing a list)
            into a comma-separated string.
            """
            
            if isinstance(val, list):
                # Already a list of strings
                return ", ".join(val)
            elif isinstance(val, str):
                # Possibly a string like "['foo', 'bar']"
                # Try literal_eval to see if it's a valid Python list
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        # It's a real list, convert to comma-separated
                        return ", ".join(parsed)
                    else:
                        # Not a list—just return the original string
                        return val
                except (SyntaxError, ValueError):
                    # It's not a parseable list, return as is
                    return val
            else:
                # Fallback: convert whatever it is to string
                return str(val)

        # 2) Normalize reactions_list so every row is a comma-separated string
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(to_comma_separated)

        # 3) Group by and aggregate
        grouped_df = (
            answers_df
            .groupby(["title", "abstract", "reactions", "reactions_list", "patient_description"], dropna=False)
            .apply(lambda grp: grp["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )

        # 4) Convert that comma-separated string (in grouped_df) back to a list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(
            lambda s: s.split(", ")
        )
        
        rerank_prompt = (
            f"Given {{patient_description}}, pick the {top_k} most applicable adverse drug reactions from the options "
            f"that are directly expressed in the following list: {{pred_reaction}}. "
            "Rank from most applicable to least applicable. "
            "Always write your answer as a list of comma-separated adverse drug reactions only and nothing else."
        )
        
        docetl_rerank_prompt = (
        """Given the following medical article, rank the following conditions in order
            of most confident to least confident that the article is describing the condition:

            {patient_description}


            Conditions: {pred_reaction}

            There may be conditions described in the article that are not in the list. Do
            not include them in the ranked list. Only focus on the conditions in the list.
        """
        )

        rerank_num_lm = grouped_df.shape[0]

        grouped_df = grouped_df.sem_map(
            docetl_rerank_prompt
        )
        

        end_t = time.time()

        grouped_df.to_csv(f"biodex_reranked_answers_{name}.csv", index=True)

        # Parse output
        known_prefixes = [
            f"Based on the patient description, the {top_k} most applicable adverse drug reactions are:\n\n",
            f"Based on the Patient_description, the {top_k} most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            f"Based on the Patient_description, the {top_k} most applicable adverse drug reactions are:\n\n",
            f"Based on the provided Patient_description, the {top_k} most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions:\n\n",
            "Here is the answer:\n\n",
            f"Here is the list of the {top_k} most applicable adverse drug reactions:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions:\n\n",
            f"Here is the list of {top_k} most applicable adverse drug reactions from the options, ranked from most applicable to least applicable:"
        ]
        def remove_known_prefixes(text: str, prefixes: list) -> str:
            """
            Removes the first matching prefix from 'text' if found in 'prefixes',
            otherwise returns text unchanged.
            """
            for prefix in prefixes:
                if text.startswith(prefix):
                    return text[len(prefix):]
            return text

        grouped_df["_map"] = grouped_df["_map"].fillna("").apply(
            lambda x: remove_known_prefixes(x, known_prefixes)
        )
        grouped_df.rename(columns={"pred_reaction": "pred_reaction_norank"}, inplace=True)
        grouped_df["pred_reaction"] = grouped_df["_map"].apply(
            lambda x: [reaction.strip() for reaction in x.split(",") if reaction.strip()]
        )

        return grouped_df, (end_t - start_t)



class parseRerankJoinCascade(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, queries_df, corpus_df):
        top_answer_df = pd.read_csv("biodex_reranked_answers.csv", index_col=0)
        
        start_t = time.time()
        known_prefixes = [
            "Based on the patient description, the 25 most applicable adverse drug reactions are:\n\n",
            "Based on the Patient_description, the 25 most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            "Based on the Patient_description, the 25 most applicable adverse drug reactions are:\n\n",
            "Based on the provided Patient_description, the 25 most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            "Here is the list of 25 most applicable adverse drug reactions:\n\n",
            "Here is the answer:\n\n",
        ]
        def remove_known_prefixes(text: str, prefixes: list) -> str:
            """
            Removes the first matching prefix from 'text' if found in 'prefixes',
            otherwise returns text unchanged.
            """
            for prefix in prefixes:
                if text.startswith(prefix):
                    return text[len(prefix):]
            return text
        
        top_answer_df["_map"] = top_answer_df["_map"].fillna("").apply(
            lambda x: remove_known_prefixes(x, known_prefixes)
        )
        
        top_answer_df["pred_reaction"] = top_answer_df["_map"].apply(
            lambda x: [reaction.strip() for reaction in x.split(",") if reaction.strip()]
        )
        top_answer_df["pred_reaction"] = top_answer_df["pred_reaction"].apply(
            lambda reactions: [f"'{r}'" for r in reactions]
        )
        end_t = time.time()

        
        print(f"top_answer_df = {top_answer_df}")

        return top_answer_df, (end_t - start_t) 
    

class checkRes(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, queries_df, corpus_df, csv_file="biodex_reranked_answers.csv"):
        start_t = time.time()
        top_answer_df = pd.read_csv(csv_file, index_col=0)
        
        def parse_list_col(x):
            """
            Try to parse x (a string) as a Python list using literal_eval.
            If parsing fails, return x unchanged.
            """
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    # If x is not a valid Python literal, just return x
                    return x
            return x
        
        # Convert string -> list for the specified columns
        for col in ["pred_reaction", "reactions_list"]:
            if col in top_answer_df.columns:
                top_answer_df[col] = top_answer_df[col].apply(parse_list_col)

        
        end_t = time.time()

        return top_answer_df, (end_t - start_t) 


class JoinCascadeDocETL(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, queries_df, corpus_df, recall_target=0.8, precision_target=0.8, range=250, name="llama"):
        # lotus.settings.
        start_t = time.time()

        # "given the {abstract} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions.",
        map_instruction = "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions."
        join_instruction = """Can the following condition be found in the following medical article?
        Medical article: {patient_description} 
        
        Condition we are looking for: {reaction}


        Determine if {reaction} is described in the medical article, considering the context and meaning beyond just the presence of individual words."""
        print(f"corpus_df = {corpus_df.shape}")

        cascade_args = CascadeArgs(
            recall_target=recall_target,
            precision_target=precision_target,
            sampling_percentage=0.00008218, #len = 500
            map_instruction=map_instruction,
            # map_examples=map_dem_df,
            cascade_IS_weight=0.9,
            cascade_IS_random_seed=42,
            cascade_IS_max_sample_range=range
        )
        answers_df, stats = queries_df.sem_join(corpus_df, join_instruction, cascade_args=cascade_args, return_stats=True)

        end_t = time.time()

        answers_df.to_csv(f"biodex_cascade_answers_for_lm_rerank_{name}.csv", index=True)
        
        print(f"Time taken: {end_t - start_t}")
        print(f"answers_df = {answers_df.columns.tolist()}")
        print(f"stats = {stats}")

        # post process for checking answers
        # convert reaction_list to string so we can groupby
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(
            lambda x: ", ".join(x)
        )
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list"])
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # convert reaction list from string back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(
            lambda x: x.split(", ")
        )

        return grouped_df, (end_t - start_t)  # qid    
    
ALL_SAMPLES = 4249
out = []
if __name__ == "__main__":
    ts = BiodexTester(n_samples=250, truncation_limit=8192)

    # LLama with our param
    ts.add_pipeline(JoinCascade(recall_target=0.9, precision_target=0.9, range=100, name="LLama-plan1"))
    
    
    # DocETL
    # ts.add_pipeline(JoinCascadeDocETL(recall_target=0.95, precision_target=0.95, range=100, name="DocETL"))


    # ts.add_pipeline(rerank(name="GPT4o-mini"))
    # ts.add_pipeline(rerank(name="GPT4o-mini-full"))
    # ts.add_pipeline(rerank(name="llama"))
    # ts.add_pipeline(rerankJoinCascade(recall_target=0.9, precision_target=0.9, range=150, top_k=25, name="GPT4o-mini"))
    # ts.add_pipeline(checkRes(csv_file="biodex_results/JoinCascade/nsamples=250_recall_target=0.7_precision_target=0.7_range=250/res.csv"))
    # ts.add_pipeline(parseRerankJoinCascade())
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=200))
    # ts.add_pipeline(JoinCascade(recall_target=0.7, precision_target=0.7, range=250))


    # RUN ALL PIPELINES
    # ---------------------------------------------------------------------
    ts.test_pipelines()

    # PRINT SUMMARY OF PIPELINE RESULTS
    # print(ts.summarize_pipeline_results("biodex_results", ts.n_samples, ["ApproxMapRetrieveFilter"]))
    print(ts.summarize_pipeline_results("biodex_results", ts.n_samples, ["JoinCascade"]))

    # for target, lat in zip(t_pairs.keys(), out):
    #     print(f"target={target}, lat={lat}")
        

    
