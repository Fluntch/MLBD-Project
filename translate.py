import pandas as pd
import time
import deepl
from tqdm import tqdm

# Load translated data
df = pd.read_csv("translated_retr.csv")

# Re-initialize DeepL
auth_key = "cb200b55-d1f4-4483-a1c6-3c8cb05fc576:fx"
translator = deepl.Translator(auth_key)

# Identify rows with translation errors
mask_error = df["messages_user_en"].apply(lambda lst: "[TRANSLATION ERROR]" in lst)
df_errors = df[mask_error].copy()

# Parse messages_user from string to list if needed
if isinstance(df_errors["messages_user"].iloc[0], str):
    import ast
    df_errors["messages_user"] = df_errors["messages_user"].apply(ast.literal_eval)

# Retry translation
def retry_translation(messages):
    results = []
    for msg in messages:
        if not msg.strip():
            results.append("")  # Skip empty messages
            continue
        if msg == "[TRANSLATION ERROR]":
            results.append("[TRANSLATION ERROR]")  # Already marked as error
            continue
        try:
            translated = translator.translate_text(msg, source_lang="DE", target_lang="EN-US").text
            results.append(translated)
        except Exception as e:
            print("⚠️ Translation error occurred. Waiting 30 seconds before retrying...")
            print(e)
            time.sleep(30)
            try:
                results.append(translated)
            except Exception:
                results.append("[TRANSLATION ERROR]")
    return results

# Progressively retry and save
output_path = "translated_retry.csv"
save_interval = 50

pbar = tqdm(total=len(df_errors), desc="Retrying failed translations")
for idx, (i, row) in enumerate(df_errors.iterrows()):
    df_errors.at[i, "messages_user_en"] = retry_translation(row["messages_user"])

    if (idx + 1) % save_interval == 0 or idx == len(df_errors) - 1:
        df.update(df_errors)
        df.to_csv(output_path, index=False)
        pbar.set_postfix(saved_rows=idx + 1)

    pbar.update(1)

pbar.close()