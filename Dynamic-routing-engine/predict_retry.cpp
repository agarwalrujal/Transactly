#include <aws/core/Aws.h>
#include <aws/core/utils/Outcome.h>
#include <aws/sagemaker-runtime/SageMakerRuntimeClient.h>
#include <aws/sagemaker-runtime/model/InvokeEndpointRequest.h>
#include <iostream>

int main() {
    Aws::SDKOptions options;
    Aws::InitAPI(options);
    {
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.region = "ap-south-1";  // ✅ Change if needed

        Aws::SageMakerRuntime::SageMakerRuntimeClient client(clientConfig);

        Aws::SageMakerRuntime::Model::InvokeEndpointRequest request;
        request.SetEndpointName("xgb-retry-predictor");  // ✅ Use your endpoint name
        request.SetContentType("application/json");

        // Sample JSON input
        Aws::StringStream ss;
        ss << R"({
            "corridor_id": "IN_EU",
            "status": "ACTIVE",
            "corridor_type": "primary",
            "success_rate_7d": 0.91,
            "latency_ms": 180,
            "cost_score": 0.65,
            "past_retry_success_rate": 0.87
        })";

        request.SetBody(Aws::MakeShared<Aws::StringStream>("Payload", ss.str()));

        auto outcome = client.InvokeEndpoint(request);

        if (outcome.IsSuccess()) {
            auto &result = outcome.GetResult();
            std::cout << "✅ Prediction: " << result.GetBody()->rdbuf() << std::endl;
        } else {
            std::cerr << "❌ Error: " << outcome.GetError().GetMessage() << std::endl;
        }
    }
    Aws::ShutdownAPI(options);
    return 0;
}
