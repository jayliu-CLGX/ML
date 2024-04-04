# src/roof_condition_insights.py
import requests
from datetime import datetime
import json
import logging
import base64
import os
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import BASE_URLS_AUTH, BASE_URLS_RCI, DATA_DIR, PDF_DIR, JSON_DIR, LOGGING_CONFIG

# Configure logging based on LOGGING_CONFIG from config.py
logging.basicConfig(**LOGGING_CONFIG)

class RoofConditionInsights:
    def __init__(self, env, api_key, api_companyid, username, password, verbose=False):
        self.env = env
        self.api_key = api_key
        self.api_companyid = api_companyid
        self.username = username
        self.password = password
        self.verbose = verbose
        self.data_dir = DATA_DIR  # From config.py
        self.pdf_dir = PDF_DIR  # From config.py
        self.json_dir = JSON_DIR  # From config.py
        self.base_urls_auth = BASE_URLS_AUTH  # From config.py
        self.base_urls_rci = BASE_URLS_RCI  # From config.py
        self._ensure_directories_exist()
        self.access_token = self.authenticate()
    def _ensure_directories_exist(self):
        for directory in [self.data_dir, self.pdf_dir, self.json_dir]:
            os.makedirs(directory, exist_ok=True)              
    
    def authenticate(self):
        """Authenticate with the server and retrieve a token."""
        auth_url = f"{self.base_urls_auth[self.env]}/Login"
        response = requests.get(auth_url, auth=(self.username, self.password))
        if response.status_code == 200:
            tokens = response.json()
            access_token = tokens["accessToken"]
            if self.verbose:
                print(f"Access token is retrieved with status code 200.")
            return access_token
        else:
            if self.verbose:
                print(f"Authentication failed with status code {response.status_code}")
            logging.error(f"Authentication failed with status code {response.status_code}")
            return None
    
    # def get_roof_condition_insights(self, address, generate_pdf=False, save_json=False):
    #     """Retrieve roof condition insights for a given address."""
    #     # if self.verbose:
    #     #     print(f"---\nRetrieve roof condition insights for address: {address}")
    #     if not self.access_token:  # Check if access_token is None before proceeding
    #         logging.error("No access token available. Authentication may have failed.")
    #         return None
    #     insights_url = f"{self.base_urls_rci[self.env]}/api/uwc/v1/roof-condition-insights/details"
    #     params = {"address": address, "generatePdf": str(generate_pdf).lower()}
    #     headers = {
    #         'Content-Type': 'application/json',
    #         'x-api-key': self.api_key,
    #         'x-api-company': self.api_companyid,
    #         'accesstoken': self.access_token, 
    #         'Authorization': f"Bearer {self.access_token}"
    #     }
    #     response = requests.get(insights_url, headers=headers, params=params)
    #     return self._handle_response(response, generate_pdf, address, save_json)
    
    
    def get_roof_condition_insights(self, address, generate_pdf=False, save_json=False, retries=1):
        """Retrieve roof condition insights for a given address, with retries on failure."""
        # if self.verbose:
        #     print(f"---\nRetrieve roof condition insights for address: {address}")
        if not self.access_token:  # Check if access_token is None before proceeding
            logging.error("No access token available. Authentication may have failed.")
            return None

        insights_url = f"{self.base_urls_rci[self.env]}/api/uwc/v1/roof-condition-insights/details"
        params = {"address": address, "generatePdf": str(generate_pdf).lower()}
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'x-api-company': self.api_companyid,
            'accesstoken': self.access_token, 
            'Authorization': f"Bearer {self.access_token}"
        }

        for attempt in range(retries):
            response = requests.get(insights_url, headers=headers, params=params)
            if response.status_code == 200:
                return self._handle_response(response, generate_pdf, address, save_json)
            # else:
        # if self.verbose:
        #     print(f"Attempt {attempt + 1} for address: {address} failed with status code {response.status_code}. Retrying...")
        # logging.warning(f"Attempt {attempt + 1} for address: {address} failed for address {address} with status code {response.status_code}. Retrying...")

        # Final attempt with no more retries, directly handle the response
        return self._handle_response(response, generate_pdf, address, save_json)


    
    def _handle_response(self, response, generate_pdf, address, save_json, skip_keys=["images","pdfContent"]):
        """Handle different HTTP status codes for the response."""
        if response.status_code == 200:
            # if self.verbose:
            #     print(f"Roof Condition Insights data is retrieved {response.status_code}: {response.reason}")
            data = response.json()
            filename = address.replace(" ", "_").replace(",", "").replace("/", "_")

            if generate_pdf and 'pdfContent' in data:
                self._save_pdf(data['pdfContent'], filename)
                
            if save_json:
                data = {k: v for k, v in data.items() if k not in skip_keys}
                json_path = os.path.join(self.json_dir, f"{filename}.json")
                self.save_json(data, json_path)
                
            return data, response.status_code
        else:
            # if self.verbose:
            #     print(f"Failed to retrieve insights with status code {response.status_code}: {response.reason}")
            # logging.error(f'Failed to retrieve insights for address "{address}" with status code {response.status_code}: {response.reason}')

            return None, response.status_code
        
    @staticmethod
    def save_json(data, filename, skip_keys=["images","pdfContent"]):
        """Save a dictionary to a JSON file, optionally skipping specified keys."""
        # data = {k: v for k, v in data.items() if k not in skip_keys}

        # Save the remaining data to a JSON file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
            # if self.verbose:
            #     print(f"Data saved to {filename}")

    def _save_pdf(self, pdf_content, filename):
        """Save base64 encoded PDF content to a file."""
        pdf_data = base64.b64decode(pdf_content)
        pdf_path = os.path.join(self.pdf_dir, f"{filename}.pdf")
        with open(pdf_path, 'wb') as file:
            file.write(pdf_data)
        # if self.verbose:
        #     print(f"PDF saved to {pdf_path}")
    
    def normalize_string(self, s):
        """Normalize string by removing special characters, whitespaces, and making it lowercase."""
        return re.sub(r'[^a-z0-9]+', '', s.lower())

    def check_condition_exists(self, conditions_list, condition_to_check):
        """
        Enhanced to check for specific variations of conditions related to tiles and shingles.
        """
        # Normalizing the input condition for comparison
        normalized_condition_to_check = self.normalize_string(condition_to_check)

        # Handling variations explicitly
        variations = {
            'tileshinglestaining': ['tilestaining', 'shinglestaining', 'tile/shinglestaining'],
            'missingtileshingles': ['missingtile', 'missingshingles', 'missingtile/shingles'],
        }

        # Check if the condition to check has predefined variations
        if normalized_condition_to_check in variations:
            check_variations = variations[normalized_condition_to_check]
        else:
            check_variations = [normalized_condition_to_check]

        for condition_dict in conditions_list:
            normalized_condition = self.normalize_string(condition_dict['condition'])
            for variation in check_variations:
                if variation in normalized_condition:
                    return True
        return False
    def calculate_roof_condition_score(self, row):
        roof_condition_dicts = row['roofCharacteristics_roofConditions']
        base_conditions = {
            "Missing Tile / Shingles": 60 if self.check_condition_exists(roof_condition_dicts, "Missing Tile / Shingles") else 0,
            "Ponding": 10 if self.check_condition_exists(roof_condition_dicts, "Ponding") else 0,
            "Partial Repair": 10 if self.check_condition_exists(roof_condition_dicts, "Partial Repair") else 0,
            "Rusting": 10 if self.check_condition_exists(roof_condition_dicts, "Rusting") else 0,
            "Structural Damage": 100 if self.check_condition_exists(roof_condition_dicts, "Structural Damage") else 0,
            "Temporary Repairs": 90 if self.check_condition_exists(roof_condition_dicts, "Temporary Repairs") else 0,
            "Tile/Shingle Staining": 30 if self.check_condition_exists(roof_condition_dicts, "Tile / Shingle Staining") else 0,
            "Zinc Staining": 10 if self.check_condition_exists(roof_condition_dicts, "Zinc Staining") else 0,
            "Tree Overhang": 20 if self.check_condition_exists(roof_condition_dicts, "Tree Overhang") else 0,
        }
        weather_conditions = {
            "Wind speed >45": 65 if row['weatherVerification_numberOfWindSpeedEventsGT45Mph'] > 0 else 0,
            "Wind speed >60": 90 if row['weatherVerification_numberOfWindSpeedEventsGT60Mph'] > 0 else 0,
            "Wind speed >80": 120 if row['weatherVerification_numberOfWindSpeedEventsGT80Mph'] > 0 else 0,
            "Hail Insight Score > 4": 30 if row['weatherVerification_hailRiskScore'] > 4 else 0,
            "Estimated current roof value < $3000": 125 if row['propertyCharacteristics_currentRoofValueEstimateWODebrisRemoval'] < 3000 else 0,
        }
        rcs_v1 = sum(base_conditions.values())
        key_triggers = [condition for condition, score in base_conditions.items() if score > 0]
        rcs_v2 = rcs_v1 + sum(weather_conditions.values())
        key_triggers += [condition for condition, score in weather_conditions.items() if score > 0]

        rcs_v1_normalized = rcs_v1 / 20
        rcs_v2_normalized = rcs_v2 / 20
        
#         # Adjusting scores outside the 1 to 10 range
#         score_boundary_v1 = ""
#         score_boundary_v2 = ""

#         if rcs_v1_normalized < 1:
#             rcs_v1_normalized = 1
#             score_boundary_v1 = "lower_bound"
#         elif rcs_v1_normalized > 10:
#             rcs_v1_normalized = 10
#             score_boundary_v1 = "upper_bound"

#         if rcs_v2_normalized < 1:
#             rcs_v2_normalized = 1
#             score_boundary_v2 = "lower_bound"
#         elif rcs_v2_normalized > 10:
#             rcs_v2_normalized = 10
#             score_boundary_v2 = "upper_bound"

        rcs_v1_clipped = max(1, min(rcs_v1_normalized, 10))
        rcs_v2_clipped = max(1, min(rcs_v2_normalized, 10))


        return pd.Series([rcs_v1_clipped, rcs_v2_clipped, key_triggers, rcs_v1_normalized, rcs_v2_normalized], 
                     index=['rci_score_v1.0', 'rci_score_v1.1', 'rci_key_triggers', 'rci_raw_score_v1.0', 'rci_raw_score_v1.1'])

    def process_insights_data(self, address, generate_pdf=False, save_json=False, retries=1):
        """Calculate roof condition scores and return processed DataFrame."""
        insights, status_code = self.get_roof_condition_insights(address, generate_pdf, save_json,  retries=retries)
        if status_code != 200:
            return pd.DataFrame({'address': [address], 'run_completed': [False], 'run_status_code':[str(int(status_code))]})

        data = pd.json_normalize(insights, sep="_")
        data['address'] = address
        data[['rci_score_v1.0', 'rci_score_v1.1', 'rci_key_triggers', 'rci_raw_score_v1.0', 'rci_raw_score_v1.1']] = data.apply(self.calculate_roof_condition_score, axis=1)
        data['run_completed'] = True
        data['run_status_code'] = str(int(status_code))
        return data


# #     ##  method 1: sequential processing
#     def process_multiple_addresses(self, addresses, generate_pdf=False, save_json=False, start_index=0):
#         """Process insights data for multiple addresses and save to a CSV with checkpoints."""
#         logging.info(f"Starting new run for {len(addresses)} addresses from index {start_index}...")
#         dfs = []
#         last_checkpoint_filename = None  # Keep track of the last saved checkpoint filename

#         for index, address in enumerate(addresses[start_index:], start=start_index):
#             try:
#                 data = self.process_insights_data(address, generate_pdf, save_json)
#                 dfs.append(data)
#                 # Log progress every 1000 addresses (changed to every 5 for demonstration)
#                 if (index + 1) % 1000 == 0 or (index + 1) == len(addresses):
#                     logging.info(f"Processed {index + 1} of {len(addresses)} addresses...")

#                     # Delete previous checkpoint if it exists
#                     if last_checkpoint_filename:
#                         try:
#                             os.remove(last_checkpoint_filename)
#                             print(f"Deleted previous checkpoint: {last_checkpoint_filename}")
#                         except OSError as e:
#                             logging.error(f"Error deleting previous checkpoint {last_checkpoint_filename}: {e}")

#                     # Save new checkpoint with month and date in the filename
#                     checkpoint_df = pd.concat(dfs, ignore_index=True)
#                     now = datetime.now().strftime("%Y%m%d_%H%M")
#                     checkpoint_filename = f'rci_data_checkpoint_{index + 1}_{now}.csv'  # Added month and date to the filename
#                     checkpoint_filepath = os.path.join(self.data_dir, 'datasets/output', checkpoint_filename)
#                     checkpoint_df.to_csv(checkpoint_filepath)
#                     print(f"Checkpoint saved to {checkpoint_filepath}")
#                     last_checkpoint_filename = checkpoint_filepath  # Update the last checkpoint filename

#             except Exception as e:
#                 logging.error(f"Failed to process address at index {index} with error: {e}")
#                 # Optionally, break or continue based on your error handling policy

#         if dfs:
#             df_roof_condition_insights = pd.concat(dfs, ignore_index=True)
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M")
#             final_csv_filepath = os.path.join(self.data_dir, 'datasets/output', f'rci_data_final_{timestamp}.csv')
#             df_roof_condition_insights.to_csv(final_csv_filepath)
#             print(f"Final RCI Data saved to {final_csv_filepath}")
#         else:
#             print("No data to save after processing.")

#         logging.info(f"Finished running for {len(addresses)} addresses.")
#         return df_roof_condition_insights if dfs else None
    # ----
    # ---
    # ---
    
#     ## method 2: parallel processing
#     def save_checkpoint(self, dfs, index, last_checkpoint_filename):
#         """Saves current progress as a checkpoint and deletes the previous one if exists."""
#         checkpoint_df = pd.concat(dfs, ignore_index=True)
#         now = datetime.now().strftime("%Y%m%d_%H%M")
#         checkpoint_filename = f'rci_data_checkpoint_{index}_{now}.parquet'
#         checkpoint_filepath = os.path.join(self.data_dir, 'datasets/output', checkpoint_filename)
#         checkpoint_df.to_parquet(checkpoint_filepath)
#         if self.verbose:
#             print(f"Checkpoint saved to {checkpoint_filepath}")

#         if last_checkpoint_filename and os.path.exists(last_checkpoint_filename):
#             os.remove(last_checkpoint_filename)
#             if self.verbose:
#                 print(f"Deleted previous checkpoint: {last_checkpoint_filename}")
        
#         return checkpoint_filepath

#     def process_address(self, address, generate_pdf, save_json, retries=1):
#         """Wrapper method for processing a single address with error handling."""
#         try:
#             # Attempt to process the address using your existing logic
#             data = self.process_insights_data(address, generate_pdf, save_json, retries)

#             return (data, None)  # Return data and None for error if successful
#         except Exception as e:
#             # Log the error and return None for data and the error
#             logging.error(f"Failed to process address {address} with error: {e}")
#             return (None, str(e))  # Ensure the second value is the error message or exception


# #     def process_multiple_addresses(self, addresses, generate_pdf=False, save_json=False, start_index=0, max_workers=5):
# #         logging.info(f"Starting new run for {len(addresses)} addresses from index {start_index}...")
# #         dfs = []
# #         last_checkpoint_filename = None

# #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
# #             future_to_index = {executor.submit(self.process_address, address, generate_pdf, save_json): i for i, address in enumerate(addresses[start_index:], start=start_index)}
            
# #             for future in as_completed(future_to_index):
# #                 index = future_to_index[future]
# #                 data, error = future.result()
                
# #                 if error:
# #                     logging.error(f"Failed to process address at index {index} with error: {error}")
# #                 else:
# #                     dfs.append(data)

# #                 if (index + 1) % 500 == 0 or (index + 1) == len(addresses):
# #                     print(f"\n\nProcessed {index + 1} / {len(addresses)} addresses...\n\n")
# #                     logging.info(f"Processed {index + 1} of {len(addresses)} addresses...")
# #                     last_checkpoint_filename = self.save_checkpoint(dfs, index + 1, last_checkpoint_filename)
        
# #         if dfs:
# #             df_roof_condition_insights = pd.concat(dfs, ignore_index=True)
# #             timestamp = datetime.now().strftime("%Y%m%d_%H%M")
# #             final_csv_filepath = os.path.join(self.data_dir, 'datasets/output', f'rci_data_final_{timestamp}.csv')
# #             df_roof_condition_insights.to_csv(final_csv_filepath, index=False)
# #             print(f"Final RCI Data saved to {final_csv_filepath}")
# #         else:
# #             print("No data to save after processing.")

# #         logging.info(f"Finished running for {len(addresses)} addresses.")
# #         return df_roof_condition_insights if dfs else None
 


#     def process_multiple_addresses(self, addresses, generate_pdf=False, save_json=False, start_index=0, process_method='sequential', max_workers=5, retries=1, check_point_count=1000):
#         """
#         Process insights data for multiple addresses with options for sequential or parallel execution.
        
#         Parameters:
#         - addresses: List of addresses to process.
#         - generate_pdf: Boolean flag to generate PDFs.
#         - save_json: Boolean flag to save JSONs.
#         - start_index: Starting index for processing addresses.
#         - process_method: 'sequential' or 'parallel' processing.
#         - max_workers: Number of workers for parallel processing.
#         """
#         logging.info(f"Starting new run for {len(addresses)} addresses from index {start_index} using {process_method} processing...")
#         dfs = []
#         last_checkpoint_filename = None

#         def process_address_wrapper(address):
#             try:
#                 data = self.process_insights_data(address, generate_pdf, save_json, retries)
#                 return data
#             except Exception as e:
#                 logging.error(f"Failed to process address {address} with error: {e}")
#                 return None

#         if process_method == 'sequential':
#             for index, address in enumerate(addresses[start_index:], start=start_index):
#                 data = process_address_wrapper(address)
#                 if not data.empty:
#                     dfs.append(data)
#                 if (index + 1) % check_point_count == 0 or (index + 1) == len(addresses):
#                     print(f"Processed {index + 1} of {len(addresses)} addresses...")
#                     logging.info(f"Processed {index + 1} of {len(addresses)} addresses...")
#                     last_checkpoint_filename = self.save_checkpoint(dfs, index + 1, last_checkpoint_filename)

#         elif process_method == 'parallel':
#             with ThreadPoolExecutor(max_workers=max_workers) as executor:
#                 future_to_address = {executor.submit(process_address_wrapper, address): address for address in addresses[start_index:]}
#                 for future in as_completed(future_to_address):
#                     address = future_to_address[future]
#                     data = future.result()
#                     if not data.empty:
#                         dfs.append(data)
#                     index = addresses.index(address)
#                     if (index + 1) % check_point_count == 0 or (index + 1) == len(addresses):
#                         print(f"Processed {index + 1} of {len(addresses)} addresses...")
#                         logging.info(f"Processed {index + 1} of {len(addresses)} addresses...")
#                         last_checkpoint_filename = self.save_checkpoint(dfs, index + 1, last_checkpoint_filename)

#         # Finalize and save data
#         if dfs:
#             df_roof_condition_insights = pd.concat(dfs, ignore_index=True)
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M")
#             final_parquet_filepath = os.path.join(self.data_dir, 'datasets/output', f'rci_data_final_{timestamp}.parquet')
#             df_roof_condition_insights.to_parquet(final_parquet_filepath)
#             logging.info(f"Final RCI Data saved to {final_parquet_filepath}")
#         else:
#             logging.info("No data to save after processing.")

#         logging.info(f"Finished running for {len(addresses)} addresses using {process_method} processing.")
#         return df_roof_condition_insights if dfs else None

    
    # ---
    # ---
    # ---

#     def save_checkpoint(self, df, index, folder_name):
#         """Saves the current chunk as a checkpoint."""
#         now = datetime.now().strftime("%Y%m%d_%H%M")
#         checkpoint_filename = f'rci_data_checkpoint_{index}_{now}.parquet'
#         checkpoint_filepath = os.path.join(self.data_dir, 'datasets/output', folder_name, checkpoint_filename)
#         df.to_parquet(checkpoint_filepath)
#         if self.verbose:
#             print(f"Checkpoint saved to {checkpoint_filepath}")
#         return checkpoint_filepath

#     def process_address(self, address, generate_pdf, save_json, retries=1):
#         """Wrapper method for processing a single address with error handling."""
#         try:
#             # Attempt to process the address using your existing logic
#             data = self.process_insights_data(address, generate_pdf, save_json, retries)

#             return (data, None)  # Return data and None for error if successful
#         except Exception as e:
#             # Log the error and return None for data and the error
#             # logging.error(f"Failed to process address {address} with error: {e}")
#             return (None, str(e))

#     def process_multiple_addresses(self, addresses, start_index=0, chunk_size=1000, generate_pdf=False, save_json=False, process_method='sequential', max_workers=5, retries=1):
#         """
#         Process insights data for multiple addresses with options for sequential or parallel execution.
        
#         Parameters:
#         - addresses: List of addresses to process.
#         - start_index: Starting index for processing addresses.
#         - chunk_size: Size of each chunk of addresses to process.
#         - generate_pdf: Boolean flag to generate PDFs.
#         - save_json: Boolean flag to save JSONs.
#         - process_method: 'sequential' or 'parallel' processing.
#         - max_workers: Number of workers for parallel processing.
#         """
#         logging.info(f"\n\n---\nStarting new run for {len(addresses)} addresses from index {start_index} using {process_method} processing...")
#                 print(f"\n\n---\nStarting new run for {len(addresses)} addresses from index {start_index} using {process_method} processing...")


#         folder_name = datetime.now().strftime("Checkpoints_%Y%m%d_%H%M")
#         folder_path = os.path.join(self.data_dir, 'datasets/output', folder_name)
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#             print(f"Created folder: {folder_path}")
#         # dfs = []

#         def process_chunk(chunk):
#             chunk_dfs = []
#             for address in chunk:
#                 data = self.process_address(address, generate_pdf, save_json, retries)
#                 if data[0] is not None:  # Only append valid data
#                     chunk_dfs.append(data[0])
#             return chunk_dfs

#         # Process addresses in chunks
#         for ind, i in enumerate(range(start_index, len(addresses), chunk_size)):
#             chunk = addresses[i:i + chunk_size]

#             if process_method == 'sequential':
#                 chunk_dfs = process_chunk(chunk)
#             elif process_method == 'parallel':
#                 with ThreadPoolExecutor(max_workers=max_workers) as executor:
#                     futures = [executor.submit(self.process_address, address, generate_pdf, save_json, retries) for address in chunk]
#                     chunk_dfs = [future.result()[0] for future in as_completed(futures) if future.result()[0] is not None]
#             else:
#                 raise ValueError("Invalid process_method. Choose 'sequential' or 'parallel'.")

#             # dfs.extend(chunk_dfs)

#             # Save checkpoint for the current chunk
#             df_chunk = pd.concat(chunk_dfs, ignore_index=True)
#             checkpoint_filepath = self.save_checkpoint(df_chunk, i + 1, folder_name)
#             logging.info(f"Chunk (ind) [{i}:{min(i+chunk_size, len(addresses))}] Processed - Success: {round(100*sum(df_chunk.run_completed)/df_chunk.shape[0], 2)}%")
#             print(f"Chunk (ind) [{i}:{min(i+chunk_size, len(addresses))}] Processed - Success: {round(100*sum(df_chunk.run_completed)/df_chunk.shape[0], 2)}%")
#             # Merge the current chunk with the previous ones
#             # if checkpoint_filepath:
#             #     dfs = [pd.read_parquet(checkpoint_filepath)]
#             #     os.remove(checkpoint_filepath)  # Remove the checkpoint file
        
#         parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
#         if not parquet_files:
#             print("No parquet files found in the folder.")
#             return None
#         dfs = []
#         for file in parquet_files:
#             file_path = os.path.join(folder_path, file)
#             df = pd.read_parquet(file_path)
#             dfs.append(df)
#         # Finalize and save data
#         if dfs:
#             df_roof_condition_insights = pd.concat(dfs, ignore_index=True)
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M")
#             final_parquet_filepath = os.path.join(folder_path, f'rci_data_final_{timestamp}.parquet')
#             df_roof_condition_insights.to_parquet(final_parquet_filepath)
#             logging.info(f"Final RCI Data saved to {final_parquet_filepath}")
#         else:
#             logging.info("No data to save after processing.")

#         logging.info(f"Finished running {len(addresses)} addresses using {process_method} - Success: {round(100*sum(df_roof_condition_insights.run_completed)/df_roof_condition_insights.shape[0], 2)}%\n---\n")
#         print(f"Finished running {len(addresses)} addresses using {process_method} - Success: {round(100*sum(df_roof_condition_insights.run_completed)/df_roof_condition_insights.shape[0], 2)}%")

#         return df_roof_condition_insights if dfs else None
# ---
    def process_address(self, address, generate_pdf, save_json, retries=1):
        """Wrapper method for processing a single address with error handling."""
        try:
            # Attempt to process the address using your existing logic
            data = self.process_insights_data(address, generate_pdf, save_json, retries)

            return (data, None)  # Return data and None for error if successful
        except Exception as e:
            # Log the error and return None for data and the error
            # logging.error(f"Failed to process address {address} with error: {e}")
            return (None, str(e))


    def process_multiple_addresses(self, addresses, start_index=0, chunk_size=1000, generate_pdf=False, save_json=False, process_method='sequential', max_workers=5, retries=1):
        """
        Process insights data for multiple addresses with options for sequential or parallel execution.
        
        Parameters:
        - addresses: List of addresses to process.
        - start_index: Starting index for processing addresses.
        - chunk_size: Size of each chunk of addresses to process.
        - generate_pdf: Boolean flag to generate PDFs.
        - save_json: Boolean flag to save JSONs.
        - process_method: 'sequential' or 'parallel' processing.
        - max_workers: Number of workers for parallel processing.
        """
        logging.info(f"\n\n---\nStarting new run for {len(addresses)} addresses from index {start_index} using {process_method} processing...")
        print(f"\n\n---\nStarting new run for {len(addresses)} addresses from index {start_index} using {process_method} processing...")

        folder_name = datetime.now().strftime("Checkpoints_%Y%m%d_%H%M")
        folder_path = os.path.join(self.data_dir, 'datasets/output', folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")

        def process_chunk(chunk):
            chunk_dfs = []
            for address in chunk:
                data = self.process_address(address, generate_pdf, save_json, retries)
                if data[0] is not None:  # Only append valid data
                    chunk_dfs.append(data[0])
            return chunk_dfs

        # Process addresses in chunks
        for ind, i in enumerate(range(start_index, len(addresses), chunk_size)):
            chunk = addresses[i:i + chunk_size]

            if process_method == 'sequential':
                chunk_dfs = process_chunk(chunk)
            elif process_method == 'parallel':
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(self.process_address, address, generate_pdf, save_json, retries) for address in chunk]
                    chunk_dfs = [future.result()[0] for future in as_completed(futures) if future.result()[0] is not None]
            else:
                raise ValueError("Invalid process_method. Choose 'sequential' or 'parallel'.")

            # Save checkpoint for the current chunk
            df_chunk = pd.concat(chunk_dfs, ignore_index=True)
            checkpoint_filepath = self.save_checkpoint(df_chunk, (i,min(i+chunk_size, len(addresses))), folder_name)
            logging.info(f"Chunk (index) [{i}:{min(i+chunk_size, len(addresses))}] Processed - Success: {round(100*sum(df_chunk.run_completed)/df_chunk.shape[0], 2)}%")
            print(f"Chunk (index) [{i}:{min(i+chunk_size, len(addresses))}] Processed - Success: {round(100*sum(df_chunk.run_completed)/df_chunk.shape[0], 2)}%")

        # Merge and save the processed data
        merged_df_filepath = self.merge_parquet_files(folder_name)

        logging.info(f"Finished running for {len(addresses)} addresses using {process_method} processing.")
        print(f"Finished running for {len(addresses)} addresses using {process_method} processing.")

        return merged_df_filepath

    def merge_parquet_files(self, folder_name):
        """Merge all parquet files in the specified folder and save the merged dataframe."""
        folder_path = os.path.join(self.data_dir, 'datasets/output', folder_name)
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            print(f"Folder Not Found: {folder_path}")

        # Get a list of all parquet files in the folder
        parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
        def sorting_key(filename):
            # Extract the value after 'checkpoint_'
            start_index = filename.find('checkpoint_') + len('checkpoint_')
            end_index = filename.find('_to_')
            value_str = filename[start_index:end_index]
            return int(value_str)

        # Sort the list using the custom sorting key
        parquet_files = sorted(parquet_files, key=sorting_key)
        if not parquet_files:
            print("No parquet files found in the folder.")
            return None

        # Read each parquet file and append it to a list
        dfs = []
        for file in parquet_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_parquet(file_path)
            dfs.append(df)

        # Merge all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)

        # Save the merged dataframe
        merged_filename = f'merged_{folder_name}.parquet'
        merged_filepath = os.path.join(self.data_dir, 'datasets/output', merged_filename)
        merged_df.to_parquet(merged_filepath)
        print(f"Merged dataframe saved to {merged_filepath}")

        return merged_filepath

    def save_checkpoint(self, df, indeces, folder_name):
        """Saves current progress as a checkpoint."""
        now = datetime.now().strftime("%Y%m%d_%H%M")
        checkpoint_filename = f'checkpoint_{indeces[0]}_to_{indeces[1]}_{now}.parquet'
        checkpoint_filepath = os.path.join(self.data_dir, 'datasets/output', folder_name, checkpoint_filename)
        df.to_parquet(checkpoint_filepath)
        print(f"Checkpoint saved to {checkpoint_filepath}")
        return checkpoint_filepath

    