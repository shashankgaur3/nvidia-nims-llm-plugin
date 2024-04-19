package com.customllm.llm;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import com.dataiku.common.rpc.ExternalJSONAPIClient;
import com.dataiku.dip.custom.PluginSettingsResolver.ResolvedSettings;
import com.dataiku.dip.llm.custom.CustomLLMClient;
import com.dataiku.dip.llm.online.LLMClient.ChatMessage;
import com.dataiku.dip.llm.online.LLMClient.CompletionQuery;
import com.dataiku.dip.llm.online.LLMClient.EmbeddingQuery;
import com.dataiku.dip.llm.online.LLMClient.SimpleCompletionResponse;
import com.dataiku.dip.llm.online.LLMClient.SimpleEmbeddingResponse;
import com.dataiku.dip.llm.promptstudio.PromptStudio.LLMStructuredRef;
import com.dataiku.dip.resourceusage.ComputeResourceUsage;
import com.dataiku.dip.resourceusage.ComputeResourceUsage.InternalLLMUsageData;
import com.dataiku.dip.resourceusage.ComputeResourceUsage.LLMUsageData;
import com.dataiku.dip.resourceusage.ComputeResourceUsage.LLMUsageType;
import com.dataiku.dss.shadelib.org.apache.http.impl.client.LaxRedirectStrategy;
import com.dataiku.dip.connections.AbstractLLMConnection.HTTPBasedLLMNetworkSettings;
import com.dataiku.dip.llm.utils.OnlineLLMUtils;
import com.dataiku.dss.shadelib.org.apache.http.impl.client.HttpClientBuilder;
import com.dataiku.dss.shadelib.org.apache.http.client.methods.HttpDelete;
import com.dataiku.dss.shadelib.org.apache.http.client.methods.HttpGet;
import com.dataiku.dss.shadelib.org.apache.http.client.methods.HttpPost;
import com.dataiku.dss.shadelib.org.apache.http.client.methods.HttpPut;
import com.dataiku.dip.utils.DKULogger;
import com.dataiku.dip.utils.JSON;
import com.dataiku.dip.utils.JF;
import com.dataiku.dip.utils.JF.ObjectBuilder;
import com.google.gson.*;

public class nvidiaNIMSPlugin extends CustomLLMClient {
    public nvidiaNIMSPlugin() {
    }

    private String endpointUrl;
    private String model;
    ResolvedSettings rs;
    private ExternalJSONAPIClient client;
    private InternalLLMUsageData usageData = new LLMUsageData();
    HTTPBasedLLMNetworkSettings networkSettings = new HTTPBasedLLMNetworkSettings();
    private int maxParallel=1;

    private static class RawChatCompletionMessage {
        String role;
        String content;
    }

    private static class RawChatCompletionChoice {
        RawChatCompletionMessage message;
    }

    private static class RawUsageResponse {
        int total_tokens;
        int prompt_tokens;
        int completion_tokens;
    }

    private static class RawChatCompletionResponse {
        List<RawChatCompletionChoice> choices;
        RawUsageResponse usage;
    }

    private static class EmbeddingResponse {
        List<EmbeddingResult> data = new ArrayList<>();
        RawUsageResponse usage;

    }

    private static class EmbeddingResult {
        double[] embedding;
    }

    public void init(ResolvedSettings settings) {

        this.rs = settings;
        endpointUrl = rs.config.get("endpoint_url").getAsString();
        model = rs.config.get("model").getAsString();
        maxParallel = rs.config.get("maxParallelism").getAsNumber().intValue();


        networkSettings.queryTimeoutMS = rs.config.get("networkTimeout").getAsNumber().intValue();
        networkSettings.maxRetries = rs.config.get("maxRetries").getAsNumber().intValue();
        networkSettings.initialRetryDelayMS = rs.config.get("firstRetryDelay").getAsNumber().longValue();
        networkSettings.retryDelayScalingFactor = rs.config.get("retryDelayScale").getAsNumber().doubleValue();

        Consumer<HttpClientBuilder> customizeBuilderCallback = (builder) -> {  
            builder.setRedirectStrategy(new LaxRedirectStrategy());  
            OnlineLLMUtils.add429RetryStrategy(builder, networkSettings);  
        };  
        
        String access_token = "Bearer " + rs.config.get("apikeys").getAsJsonObject().get("api_key").getAsString();;
        // TODO: Manage all AuthN/Z
        client = new ExternalJSONAPIClient(endpointUrl, null, true, null, customizeBuilderCallback)  {
            @Override
            protected HttpGet newGet(String path) {
                HttpGet get = new HttpGet(path);
                setAdditionalHeadersInRequest(get);
                get.addHeader("Content-Type", "application/json");
                get.addHeader("Authorization", access_token);
                return get;
            }

            @Override
            protected HttpPost newPost(String path) {
                HttpPost post = new HttpPost(path);
                setAdditionalHeadersInRequest(post);
                post.addHeader("Content-Type", "application/json");
                post.addHeader("Authorization", access_token);
                return post;

            }

            @Override
            protected HttpPut newPut(String path) {
                throw new IllegalArgumentException("unimplemented");
            }

            @Override
            protected HttpDelete newDelete(String path) {
                throw new IllegalArgumentException("unimplemented");
            }

        };

    }

    public int getMaxParallelism() {
        int maxParallel = rs.config.get("maxParallelism").getAsNumber().intValue();
        return maxParallel;
    }

    public synchronized List<SimpleCompletionResponse> completeBatch(List<CompletionQuery> completionQueries)
            throws IOException {
        List<SimpleCompletionResponse> ret = new ArrayList<>();
        for (CompletionQuery query : completionQueries) {

            long before = System.currentTimeMillis();
            SimpleCompletionResponse scr = null;

            logger.info("Chat Complete: " + JSON.json(query));
            scr = chatComplete(model, query.messages, query.settings.maxOutputTokens, query.settings.temperature,
                    query.settings.topP, query.settings.stopSequences);
            //scr.estimatedCost = (OpenAIPricing.getOpenAIPromptCostPer1KTokens(model) * scr.promptTokens
              //      + OpenAIPricing.getOpenAICompletionCostPer1KTokens(model) * scr.completionTokens) / 1000;

            usageData.totalComputationTimeMS += (System.currentTimeMillis() - before);
            usageData.totalPromptTokens += scr.promptTokens;
            usageData.totalCompletionTokens += scr.completionTokens;
            //usageData.estimatedCostUSD += scr.estimatedCost;

            ret.add(scr);
        }

        return ret;
    }

    public List<SimpleEmbeddingResponse> embedBatch(List<EmbeddingQuery> queries) 
            throws IOException {
        
        List<SimpleEmbeddingResponse> ret = new ArrayList<>();

        for (EmbeddingQuery query : queries) {
            long before = System.currentTimeMillis();

            logger.info("Chat Embed: " + JSON.json(query));

            SimpleEmbeddingResponse ser = null;
            ser = embed(model,query.text);

            //ser.estimatedCost = (OpenAIPricing.getOpenAIEmbeddingCostPer1KTokens(model) * scr.promptTokens) / 1000;

            usageData.totalComputationTimeMS += (System.currentTimeMillis() - before);
            usageData.totalPromptTokens += ser.promptTokens;
            //usageData.estimatedCostUSD += ser.estimatedCost;

            ret.add(ser);
        }
        return ret;
    }

    public ComputeResourceUsage getTotalCRU(LLMUsageType usageType, LLMStructuredRef llmRef) {
        ComputeResourceUsage cru = new ComputeResourceUsage();
        cru.setupLLMUsage(usageType, llmRef.connection, llmRef.type.toString());
        cru.llmUsage.setFromInternal(this.usageData);
        return cru;
    }

    public SimpleCompletionResponse chatComplete(String model, List<ChatMessage> messages, Integer maxTokens,
            Double temperature, Double topP, List<String> stopSequences) throws IOException {
        ObjectBuilder ob = JF.obj();
        ob.with("model", model);

        JsonArray jsonMessages = new JsonArray();

        messages.forEach(m -> {
            jsonMessages.add(JF.obj().with("role", m.role).with("content", m.getText()).get());
        });
        ob.with("messages", jsonMessages);

        if (maxTokens != null) {
            ob.with("max_tokens", maxTokens);
        }
        if (temperature != null) {
            ob.with("temperature", temperature);
        }

        if (topP != null) {
            ob.with("top_p", topP);
        }

        if (stopSequences != null && stopSequences.size() > 0) {
            JsonArray arr = new JsonArray();
            stopSequences.forEach(s -> {
                arr.add(s);
            });
            ob.with("stop", arr);
        }

        logger.info("Raw Chat chat completion: " + JSON.pretty(ob.get()));


        RawChatCompletionResponse rcr = client.postObjectToJSON(endpointUrl, networkSettings.queryTimeoutMS,
                RawChatCompletionResponse.class, ob.get());

        if (rcr.choices == null || rcr.choices.size() == 0) {
            throw new IOException("Chat did not respond with valid completion");
        }

        SimpleCompletionResponse ret = new SimpleCompletionResponse();

        ret.text = rcr.choices.get(0).message.content;
        ret.promptTokens = rcr.usage.prompt_tokens;
        ret.completionTokens = rcr.usage.completion_tokens;
        return ret;
    }

    public SimpleEmbeddingResponse embed(String model, String text) 
    throws IOException {
        
        String input_type = "query";
        ObjectBuilder ob = JF.obj().with("input", text).with("model", model).with("input_type",input_type);

        logger.info("raw embedding query: " + JSON.json(ob.get()));
        EmbeddingResponse rer = client.postObjectToJSON(endpointUrl, networkSettings.queryTimeoutMS,EmbeddingResponse.class, ob.get());
        logger.info("raw embedding response: " + JSON.json(rer));

        if (rer.data.size() != 1) {
            throw new IOException("Chat did not respond with valid embeddings");
        }

        SimpleEmbeddingResponse ret = new SimpleEmbeddingResponse();
        ret.embedding = rer.data.get(0).embedding;
        ret.promptTokens = rer.usage.total_tokens;
        return ret;
    }

    private static DKULogger logger = DKULogger.getLogger("dku.llm.customplugin");
}