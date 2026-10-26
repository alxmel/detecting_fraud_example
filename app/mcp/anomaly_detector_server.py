import json

import boto3
from fastmcp import FastMCP

mcp = FastMCP("anomaly")

client = boto3.client("sagemaker-runtime", region_name="us-east-1")

@mcp.tool()
def get_anomaly(payload=None) -> str:
    """ Queries a trained custom Anomaly Detection model Endpoint to flag a claim as anomalous or not. Payload is optional if provided, if not fall back to stock payload. """

    if not payload:
        payload = [
            {"CLAIM_ID": 1001, "CLAIM_DOLLAR_AMOUNT_PAID": 195.33, "CLAIM_SERVICES_FULL_SUBMISSION": 244.16, "n_lines": 3, "n_services": 6, "n_examinations": 2, "PATIENT_ID": 2001, "DOCTOR_ID": 3001, "PRACTICE_ID": 4001, "PRIMARY_EXAMINATION_CODE": "A1"},
            {"CLAIM_ID": 1002, "CLAIM_DOLLAR_AMOUNT_PAID": 627.29, "CLAIM_SERVICES_FULL_SUBMISSION": 1017.11, "n_lines": 7, "n_services": 42, "n_examinations": 2, "PATIENT_ID": 2002, "DOCTOR_ID": 3002, "PRACTICE_ID": 4002, "PRIMARY_EXAMINATION_CODE": "A1"},
            {"CLAIM_ID": 1003, "CLAIM_DOLLAR_AMOUNT_PAID": 108.16, "CLAIM_SERVICES_FULL_SUBMISSION": 108.16, "n_lines": 2, "n_services": 2, "n_examinations": 2, "PATIENT_ID": 2003, "DOCTOR_ID": 3003, "PRACTICE_ID": 4003, "PRIMARY_EXAMINATION_CODE": "A1"},
            {"CLAIM_ID": 1004, "CLAIM_DOLLAR_AMOUNT_PAID": 25000.00, "CLAIM_SERVICES_FULL_SUBMISSION": 31000.00, "n_lines": 20, "n_services": 300, "n_examinations": 12, "PATIENT_ID": 2004, "DOCTOR_ID": 3004, "PRACTICE_ID": 4004, "PRIMARY_EXAMINATION_CODE": "A1"},
            {"CLAIM_ID": 1005, "CLAIM_DOLLAR_AMOUNT_PAID": 792.10, "CLAIM_SERVICES_FULL_SUBMISSION": 1017.80, "n_lines": 13, "n_services": 156, "n_examinations": 12, "PATIENT_ID": 2005, "DOCTOR_ID": 3005, "PRACTICE_ID": 4005, "PRIMARY_EXAMINATION_CODE": "A1"}
        ]


    # The following is the Amazon SageMaker A.I. Model and Endpoint information
    response = client.invoke_endpoint(
        EndpointName="<sagemaker-endpoint-name>",
        InferenceComponentName="<sagemaker-inference-component-name>",
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read())

    flagged = []
    for prediction in result["predictions"]:
        if prediction["flagged"]:
            flagged.append({
                "Claim ID": prediction["CLAIM_ID"],
                "Doctor ID": prediction["DOCTOR_ID"],
            })
    return json.dumps(flagged) if flagged else "No claims flagged!"


# Run the MCP server;
if __name__ == "__main__":
    mcp.run()
