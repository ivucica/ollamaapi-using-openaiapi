package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"io/ioutil"
	"log"
	"net/http"
	"time"

	"badc0de.net/pkg/flagutil"
	"badc0de.net/pkg/ollamaapi-using-openaiapi/ollamaapi"

	openai "github.com/openai/openai-go"
	openaioption "github.com/openai/openai-go/option"
)

var (
	openaiAPIKey   = flag.String("openai_api_key", "", "OpenAI API key")
	openaiEndpoint = flag.String("openai_endpoint", "http://127.0.0.1:5001/v1/", "OpenAI API endpoint (default for local LM Studio: http://127.0.0.1:5000/v1/)")

	listenAddr      = flag.String("addr", "127.0.0.1:11435", "HTTP listen address serving the enabled APIs. Ollama API defaults to 11434.")
	modelMap        = flag.String("model_map", `{"deepseek-coder:1.3b-base":"deepseek-coder-1.3b-base","mistral:instruct":"mlx-community/qwen2.5-14b-instruct"}`, "Model map converting Ollama API model names to OpenAI API model names. Key specifies which model name is accepted via the API call, and value specifies which model name will be used when talking to the OpenAI-style API.")
	reverseModelMap = flag.String("reverse_model_map", ``, "Optional: Reverse model map converting OpenAI API model names to Ollama API model names. Key specifies which model name is given to us in return values of the API calls by the OpenAI-style API, and value specifies which model name will be used when returning values in the Ollama-style API responses. Passing an empty value uses autogenerated reverse map (which may be a problem if the OpenAPI-style model names are not unique). Example: "+`{"deepseek-coder-1.3b-base":"deepseek-coder:1.3b-base","mlx-community/qwen2.5-14b-instruct":"mistral:instruct"}`)
)

func init() {
	flagutil.Parse()
}

func LogReq(req *http.Request, includeBody bool) {
	body := ""
	if includeBody && req.Body != nil {
		// Read the request body
		bodyBytes, readErr := ioutil.ReadAll(req.Body)
		if readErr != nil {
			log.Printf("Error reading request body: %s", readErr)
		} else {
			body = string(bodyBytes)
			// Restore the request body so it can be read again
			req.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))
		}
	}
	log.Printf("Request: %s %s %s", req.Method, req.URL, body)
}

// Note: includeBody=true breaks streaming because it accumulates the entire
// response body in memory before doing anything with it.
//
// This could be fixed in the future.
func LogRes(res *http.Response, err error, duration time.Duration, includeBody bool) {
	if err != nil {
		log.Printf("Error: %s", err)
	} else {
		body := ""
		if includeBody && res.Body != nil {
			// Read the response body
			bodyBytes, readErr := ioutil.ReadAll(res.Body)
			if readErr != nil {
				log.Printf("Error reading response body: %s", readErr)
			} else {
				body = string(bodyBytes)
				// Restore the response body so it can be read again
				res.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))
			}
		}

		log.Printf("Response: %s %s %s %s", res.Status, res.Proto, duration, body)
	}
}

func Logger(includeReqBody bool, includeRespBody bool) openaioption.Middleware {

	return func(req *http.Request, next openaioption.MiddlewareNext) (res *http.Response, err error) {
		// Before the request
		start := time.Now()
		LogReq(req, includeReqBody)

		// Forward the request to the next handler
		res, err = next(req)

		// Handle stuff after the request
		end := time.Now()
		LogRes(res, err, end.Sub(start), includeRespBody)

		return res, err
	}
}

func main() {
	// includeReqBody determines whether the request body should be logged.
	const includeReqBody = true
	// includeRespBody determines whether the response body should be logged.
	// Note: includeRespBody=true breaks streaming because it accumulates the
	// entire response body in memory before doing anything with it.
	const includeRespBody = false

	// construct OpenAI client
	client := openai.NewClient(
		openaioption.WithBaseURL(*openaiEndpoint),
		openaioption.WithAPIKey(*openaiAPIKey),
		openaioption.WithMiddleware(Logger(includeReqBody, includeRespBody)),
	)

	modelMapMap := map[string]string{}
	if len(*modelMap) > 0 {
		if err := json.Unmarshal([]byte(*modelMap), &modelMapMap); err != nil {
			log.Fatalf("Error parsing model map: %s", err)
		}
	}
	reverseModelMapMap := map[string]string{}
	if len(*reverseModelMap) > 0 {
		if err := json.Unmarshal([]byte(*reverseModelMap), &reverseModelMapMap); err != nil {
			log.Fatalf("Error parsing reverse model map: %s", err)
		}
	}

	// create a new Ollama API handler
	handler := ollamaapi.New(
		ollamaapi.WithOpenAI(client),
		ollamaapi.WithModelMap(modelMapMap),
		ollamaapi.WithReverseModelMap(reverseModelMapMap),
	)

	// start the HTTP server
	log.Printf("Listening on %s", *listenAddr)
	log.Fatal(http.ListenAndServe(*listenAddr, handler))
}
