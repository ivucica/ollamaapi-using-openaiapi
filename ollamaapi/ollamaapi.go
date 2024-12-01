package ollamaapi

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	openai "github.com/openai/openai-go"
)

type Handler struct {
	OpenAI          *openai.Client
	ModelMap        map[string]string
	ReverseModelMap map[string]string

	mux *mux.Router
}

// Functional options are functions that will be exefcuted to apply changes
// to a Handler. They take a handler that they will apply options to, and are
// returned by the With* functions.
type HandlerOption func(*Handler)

// WithOpenAI sets the OpenAI client on a Handler.
func WithOpenAI(client *openai.Client) HandlerOption {
	return func(h *Handler) {
		h.OpenAI = client
	}
}

// WithModelMap allows overriding which model to use for a given model name.
//
// Also populates the ReverseModelMap with the reverse mapping, which might
// be broken in case the right-hand-side is not unique.
func WithModelMap(modelMap map[string]string) HandlerOption {
	return func(h *Handler) {
		h.ModelMap = modelMap
		h.ReverseModelMap = make(map[string]string, len(modelMap))
		for k, v := range modelMap {
			h.ReverseModelMap[v] = k
		}
	}
}

// WithReverseModelMap creates a reverse model map from the given model map.
// That is: when a response is received, this will be used to map the model
// name back to the original model name.
//
// If an empty map is passed, the ReverseModelMap will not be overwritten,
// allowing for use of WithModelMap if a reverse is not explicitly specified.
func WithReverseModelMap(reverseModelMap map[string]string) HandlerOption {
	return func(h *Handler) {
		if len(reverseModelMap) == 0 {
			return
		}
		h.ReverseModelMap = reverseModelMap
	}
}

// New creates a new Handler with the given options applied.
func New(options ...HandlerOption) *Handler {
	h := &Handler{}
	for _, option := range options {
		option(h)
	}

	h.mux = mux.NewRouter()
	h.mux.HandleFunc("/api/generate", h.Generate)
	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mux.ServeHTTP(w, r)
}

type GenerateRequest struct {
	// The prompt to generate a response for.
	Prompt string `json:"prompt"`
	// Required: The model name.
	//
	// From doc:
	// Model names follow a `model:tag` format, where `model` can have an
	// optional namespace such as `example/model`. Some examples are
	// `orca-mini:3b-q4_1` and `llama3:70b`. The tag is optional and, if not
	// provided, will default to `latest`. The tag is used to identify a
	// specific version.
	//
	// However, please note that the actual model name will at this time be
	// passed verbatim to the underlying OpenAI API (whether LM Studio or
	// otherwise). The underlying OpenAI API might be able to handle an unset
	// value; for example, LM Studio will be able to work without a passed API
	// as long as exactly a single model is loaded.
	Model string `json:"model"`

	// The text after the model response.
	//
	// Example: for codellama:code, assuming prompt `def compute_gcd(a, b):`,
	// this could be `    return result`.
	Suffix *string `json:"suffix,omitempty"`

	// A list of base64-encoded images (for multimodal models such as
	// `llava`).
	Images []string `json:"images,omitempty"`

	// Advanced parameters. Many might not be passed on.

	// The format to return the response in. Currently the only accepted value
	// is `json`, or it should be omitted to get back a plain text response.
	//
	// This is not passed to the OpenAI API at this time.
	Format *string `json:"format,omitempty"`
	// Additional model parameters listed in the documentation for the
	// Modelfile. (Currently not passed. Type not verified.)
	//
	// Example:
	// - `temperature` (float): The temperature to use for sampling.
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	// System message to pass to the model. (For Ollama, would be defined in
	// the Modelfile.)
	System *string `json:"system,omitempty"`
	// The prompt template to use. (For Ollama, would be defined in the
	// Modelfile.)
	Template *string `json:"template,omitempty"`
	// The context parameter returned from a previous request to `/generate`;
	// this can be used to keep a short conversational memory. (Not supported
	// by the proxy at this time.)
	Context *string `json:"context,omitempty"`
	// If `false`, the response will be returned as a single response object,
	// rather than a stream of objects. (Not supported by the proxy at this
	// time. Always requesting `false` for now.)
	//
	// Request handler will set this to true.
	Stream *bool `json:"stream,omitempty"`
	// If `true`, no formatting will be applied to the response. You may choose
	// to use the `raw` parameter if you are specifying a full templated prompt
	// in your request.
	Raw *bool `json:"raw,omitempty"`
	// Controls how long the model will stay loaded in the memory following the
	// request (default 5m in Ollama). (Not supported by the proxy at this
	// time.)
	KeepAlive *string `json:"keep_alive,omitempty"`
}

// GenerateResponse is the response object for the Generate endpoint of Ollama
// API.
//
// To calculate how fast the response is generated in tokens per second
// (token/s), divide `eval_count` / `eval_duration` * `10^9`.
//
// If `stream` is set to `false`, the response will be a single object.
type GenerateResponse struct {
	// The model name.
	Model string `json:"model"`
	// The time the response was created.
	CreatedAt string `json:"created_at"`
	// The response from the model.
	//
	// The full response if the response was not streamed.
	Response string `json:"response"`
	// Whether the response is done.
	Done bool `json:"done"`

	// Additional fields.

	// Time spent generating the response.
	TotalDuration *int64 `json:"total_duration,omitempty"`
	// Time spent loading the model.
	LoadDuration *int64 `json:"load_duration,omitempty"`
	// Number of tokens in the prompt.
	PromptEvalCount *int64 `json:"prompt_eval_count,omitempty"`
	// Time spent evaluating the prompt.
	PromptEvalDuration *int64 `json:"prompt_eval_duration,omitempty"`
	// Number of tokens in the response.
	EvalCount *int64 `json:"eval_count,omitempty"`
	// Time spent generating the response.
	EvalDuration *int64 `json:"eval_duration,omitempty"`
	// An encoding of the conversation used in this response, this can be sent
	// in the next request to keep a conversational memory.
	Context []int64 `json:"context,omitempty"`
	// DoneReason contains the reason the response is done. It can be, for
	// example, `stop`.
	//
	// OpenAPI Go client says this is the following:
	//
	// The reason the model stopped generating tokens. This will be `stop` if
	// the model hit a natural stop point or a provided stop sequence, `length`
	// if the maximum number of tokens specified in the request was reached,
	// `content_filter` if content was omitted due to a flag from our content
	// filters, `tool_calls` if the model called a tool, or `function_call`
	// (deprecated) if the model called a function.
	DoneReason *string `json:"done_reason,omitempty"`
}

// Generate handles a request for the /api/generate endpoint. It requires the
// GenerateRequest to be passed in the request body in JSON format.
func (h *Handler) Generate(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 600*time.Second)
	defer cancel()

	// Verify the request method is a POST.
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// No need to check content type. We will assume it is JSON.

	// Create the request object.
	req := &GenerateRequest{}
	// Decode the request body into the request object.
	err := json.NewDecoder(r.Body).Decode(req)
	if err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	// Print the request object for debugging (reencoded into JSON).
	reqBody := new(bytes.Buffer)
	err = json.NewEncoder(reqBody).Encode(req)
	if err != nil {
		log.Printf("Error encoding request: %s", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
	log.Printf("Request: %s %s %s", r.Method, r.URL, reqBody.String())

	// If user did not set some of the values that default to true, set them
	// now. This is why we have ptrs instead of the concrete values.
	if req.Stream == nil {
		t := true
		req.Stream = &t
	}

	// Call the OpenAI API.
	//
	// TODO: write actual streaming support
	if !(req.Stream == nil || *req.Stream) {
		h.generateUnstreamed(ctx, w, req)
	} else {
		h.generateStreamed(ctx, w, req)
	}
}

func (h *Handler) reqCommonTranslate(req *GenerateRequest) openai.ChatCompletionNewParams {

	print(req.Prompt)
	println()
	return openai.ChatCompletionNewParams{
		Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(req.Prompt),
		}),
		// Seed: openai.Int(1),
		Model: openai.F(h.OpenAIModelForOllamaModel(req.Model)),
		// Temperature: ...
		// MaxTokens: ...
		// etc
	}
}

func (h *Handler) generateUnstreamed(ctx context.Context, w http.ResponseWriter, req *GenerateRequest) {
	resp, err := h.OpenAI.Chat.Completions.New(ctx, h.reqCommonTranslate(req))
	if err != nil {
		log.Printf("Error generating response: %s", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	if len(resp.Choices) == 0 {
		http.Error(w, "No response", http.StatusBadGateway)
		return
	}

	res := respCommonTranslate(*resp)
	res.Model = h.OllamaModelForOpenAIModel(res.Model)

	// Encode the response object into a temporary buffer so we can log
	// it. Otherwise we could just json.NewEncoder(w).Encode(res).
	buf := new(bytes.Buffer)
	err = json.NewEncoder(buf).Encode(res)
	if err != nil {
		log.Printf("Error encoding response: %s", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Log the response object.
	log.Printf("Sending response: %+v", buf.String())

	// Write the response object from the buffer to the response body.
	io.Copy(w, buf)

	// Done.
}

func (h *Handler) generateStreamed(ctx context.Context, w http.ResponseWriter, req *GenerateRequest) {
	stream := h.OpenAI.Chat.Completions.NewStreaming(ctx, h.reqCommonTranslate(req))

	acc := openai.ChatCompletionAccumulator{}

	for stream.Next() {
		chunk := stream.Current()
		acc.AddChunk(chunk)

		// When this fires, the current chunk value will not contain content data
		if content, ok := acc.JustFinishedContent(); ok {
			println("Content stream finished:", content)
			println()
		}
		if tool, ok := acc.JustFinishedToolCall(); ok {
			println("Tool call stream finished:", tool.Index, tool.Name, tool.Arguments)
			println()
		}
		if refusal, ok := acc.JustFinishedRefusal(); ok {
			println("Refusal stream finished:", refusal)
			println()
		}

		// It's best to use chunks after handling JustFinished events
		if len(chunk.Choices) > 0 {
			//print(evt.Choices[0].Message.Content)
			res := respCommonTranslate(chunk)

			res.Model = h.OllamaModelForOpenAIModel(res.Model)

			// Encode the response object into a temporary buffer so we can log
			// it. Otherwise we could just json.NewEncoder(w).Encode(res).
			buf := new(bytes.Buffer)
			err := json.NewEncoder(buf).Encode(res)
			if err != nil {
				log.Printf("Error encoding response: %s", err)
				http.Error(w, "Internal server error", http.StatusInternalServerError)
				return
			}

			// Log the response object.
			log.Printf("Streaming response: %+v", buf.String())

			// Write the response object from the buffer to the response body.
			if _, err := io.Copy(w, buf); err != nil {
				log.Printf("Error writing response: %s", err)
				http.Error(w, "Internal server error", http.StatusInternalServerError)
				return
			}

			// Add a newline since Ollama does that too.
			// No need: json.NewEncoder does this already.
			//w.Write([]byte("\n"))

			if wf, ok := w.(http.Flusher); ok {
				wf.Flush()
			}
		}
	}

	// After the stream is finished, acc can be used like a ChatCompletion
	//_ = acc.Choices[0].Message.Content
	println("Total Tokens:", acc.Usage.TotalTokens)
	println("Finish Reason:", acc.Choices[0].FinishReason)

}

func respCommonTranslate[T openai.ChatCompletion | openai.ChatCompletionChunk](resp T) *GenerateResponse {

	choices := getChoices(resp)
	// Create the response object.
	res := &GenerateResponse{
		Model:     getModel(resp), // corrected in caller (no access to h here)
		CreatedAt: time.Now().Format(time.RFC3339),
		Response:  choices[0].Message.Content,
		Done:      false,
	}
	if choices[0].FinishReason.IsKnown() {
		res.DoneReason = ptrString(string(choices[0].FinishReason))
		res.Done = true
	}

	usage, hasUsage := getUsage(resp)
	// Check if usage is set to null, which indicates non-final chunk, an error,
	// or that the usage is not included in the response because it was not
	// requested via stream_options["include_usage"] == true.
	if hasUsage {
		res.PromptEvalCount = ptrInt64(usage.PromptTokens)
		res.EvalCount = ptrInt64(usage.TotalTokens)
	}

	// if set, update 'created_at' using 'created', which is a unix timestamp in
	// seconds.
	if created := getCreated(resp); created > 0 {
		res.CreatedAt = time.Unix(created, 0).Format(time.RFC3339)
	}

	return res
}

func ptrString(s string) *string { return &s }
func ptrInt64(i int64) *int64    { return &i }

func getModel[T openai.ChatCompletion | openai.ChatCompletionChunk](resp T) string {
	switch v := any(resp).(type) {
	case openai.ChatCompletion:
		return v.Model
	case openai.ChatCompletionChunk:
		return v.Model
	default:
		return ""
	}
}

// Returns v.Usage and whether v.Usage is actually null.
func getUsage[T openai.ChatCompletion | openai.ChatCompletionChunk](resp T) (openai.CompletionUsage, bool) {
	switch v := any(resp).(type) {
	case openai.ChatCompletion:
		return v.Usage, v.JSON.Usage.IsNull()
	case openai.ChatCompletionChunk:
		// v.Usage will only be set if the chunk is the last one in the stream,
		// and stream_options["include_usage"] is set to true.
		//
		// Otherwise it is set to null.
		return v.Usage, v.JSON.Usage.IsNull()
	default:
		return openai.CompletionUsage{}, true
	}
}

func getCreated[T openai.ChatCompletion | openai.ChatCompletionChunk](resp T) int64 {
	switch v := any(resp).(type) {
	case openai.ChatCompletion:
		return v.Created
	case openai.ChatCompletionChunk:
		return v.Created
	default:
		return 0
	}
}

// Note: It would make more sense to generate a streaming response if we have to
// choose just one or the other. However, this is fine for the initial attempt.
func getChoices[T openai.ChatCompletion | openai.ChatCompletionChunk](resp T) []openai.ChatCompletionChoice {
	switch v := any(resp).(type) {
	case openai.ChatCompletion:
		return v.Choices
	case openai.ChatCompletionChunk:
		choices := make([]openai.ChatCompletionChoice, len(v.Choices))
		for i, choice := range v.Choices {
			choices[i] = openai.ChatCompletionChoice{
				Index: choice.Index,
				Message: openai.ChatCompletionMessage{
					Content: choice.Delta.Content,
					Refusal: choice.Delta.Refusal,
					Role:    openai.ChatCompletionMessageRole(string(choice.Delta.Role)), // TODO: check if this is a correct enum->enum mapping
				},
				FinishReason: openai.ChatCompletionChoicesFinishReason(string(choice.FinishReason)), // TODO: check if this is a correct enum->enum mapping
			}
		}
		return choices
	default:
		return nil
	}
}

// OllamaModelForOpenAIModel returns the Ollama model name for the given OpenAI
// model name. If the model name is not found, the original model name is
// returned.
func (h *Handler) OllamaModelForOpenAIModel(model string) string {
	if m, ok := h.ReverseModelMap[model]; ok {
		return m
	}
	return model
}

// OpenAIModelForOllamaModel returns the OpenAI model name for the given Ollama
// model name. If the model name is not found, the original model name is
// returned.
func (h *Handler) OpenAIModelForOllamaModel(model string) string {
	if m, ok := h.ModelMap[model]; ok {
		return m
	}
	return model
}
