# CS611_MLE_Group_Project
This repository is for Group 9 for CS611 Machine Learning Engineering.

Your data source is located in **Google Cloud Storage**. To work with the notebooks, you'll first need to read the data from the cloud.

---

### Cloud Service & Group Information
* **Cloud Service:** Google Cloud Platform
* **Google Group:** CS611\_MLE\_Group\_9 (If you haven't joined, please contact Junqi.)

---

### **IMPORTANT: PLEASE FOLLOW THESE STEPS BEFORE PROCEEDING!**

---

### If Running in Your Local Machine, Please Follow These Steps:

1.  **Install Google Cloud CLI:**
    * Visit the official installation guide: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)

2.  **Get Authentication:**
    * Open your local terminal and run the following command:
        ```bash
        gcloud auth application-default login
        ```

3.  **Copy Credential Path:**
    * After successful authentication by GCP, you'll be assigned credentials. Their path will be displayed in your terminal; **copy this path**.

4.  **Build Docker Image:**
    * Return to your IDE and open a new terminal.
    * Create a new Docker image by running:
        ```bash
        docker build -t <your_customised_image_name> .
        ```
        *(Replace `<your_customised_image_name>` with a name of your choice.)*

5.  **Run Docker Image with Credentials:**
    * To run the image, execute the following command:
        ```bash
        docker run -it --rm -p 8890:8890 -v "<your_credential_path>:/app/.config/gcloud:ro" -e GOOGLE_APPLICATION_CREDENTIALS="/app/.config/gcloud/application_default_credentials.json" <your_customised_image_name> jupyter lab --ip=0.0.0.0 --port=8890 --no-browser --allow-root --NotebookApp.token=''
        ```
        *(Replace `<your_credential_path>` with the path you copied in step 3, and `<your_customised_image_name>` with the name you used in step 4.)*

6.  **Launch Jupyter Notebook:**
    * You can now click the URL displayed in your terminal to launch the Jupyter notebook and begin your ETL process from Google Cloud Storage.