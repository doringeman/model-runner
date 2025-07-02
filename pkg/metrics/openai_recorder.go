package metrics

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/docker/model-runner/pkg/inference"
	"github.com/docker/model-runner/pkg/inference/backends/llamacpp"
	"github.com/docker/model-runner/pkg/logging"
	"github.com/docker/model-runner/pkg/runnermap"
)

type responseRecorder struct {
	http.ResponseWriter
	body       *bytes.Buffer
	statusCode int
}

func (rr *responseRecorder) Write(b []byte) (int, error) {
	rr.body.Write(b)
	return rr.ResponseWriter.Write(b)
}

func (rr *responseRecorder) WriteHeader(statusCode int) {
	rr.statusCode = statusCode
	rr.ResponseWriter.WriteHeader(statusCode)
}

func (rr *responseRecorder) Flush() {
	if flusher, ok := rr.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

type RequestResponsePair struct {
	ID         string    `json:"id"`
	Model      string    `json:"model"`
	Method     string    `json:"method"`
	URL        string    `json:"url"`
	Request    string    `json:"request"`
	Response   string    `json:"response"`
	Timestamp  time.Time `json:"timestamp"`
	StatusCode int       `json:"status_code"`
	UserAgent  string    `json:"user_agent,omitempty"`
}

type ModelData struct {
	Config  inference.BackendConfiguration `json:"config"`
	Records []*RequestResponsePair         `json:"records"`
}

type OpenAIRecorder struct {
	log     logging.Logger
	records *runnermap.Map[*ModelData]
	m       sync.RWMutex
}

func NewOpenAIRecorder(log logging.Logger, runnerMapNormalizeFn func(string) string) *OpenAIRecorder {
	return &OpenAIRecorder{
		log:     log,
		records: runnermap.New[*ModelData](runnerMapNormalizeFn),
	}
}

func (r *OpenAIRecorder) SetConfigForModel(runner runnermap.Key, config *inference.BackendConfiguration) {
	if config == nil {
		r.log.Warnf("SetConfigForModel called with nil config for model %s", runner.Model)
		return
	}

	r.m.Lock()
	defer r.m.Unlock()

	if _, ok := r.records.Get(runner); !ok {
		r.records.Set(runner, &ModelData{
			Records: make([]*RequestResponsePair, 0, 10),
			Config:  inference.BackendConfiguration{},
		})
	}

	rr, _ := r.records.Get(runner)
	rr.Config = *config
}

func (r *OpenAIRecorder) RecordRequest(runner runnermap.Key, req *http.Request, body []byte) string {
	r.m.Lock()
	defer r.m.Unlock()

	recordID := fmt.Sprintf("%s_%d", runner.Model, time.Now().UnixNano())

	record := &RequestResponsePair{
		ID:        recordID,
		Model:     runner.Model,
		Method:    req.Method,
		URL:       req.URL.Path,
		Request:   string(body),
		Timestamp: time.Now(),
		UserAgent: req.UserAgent(),
	}

	if _, ok := r.records.Get(runner); !ok {
		r.records.Set(runner, &ModelData{
			Records: make([]*RequestResponsePair, 0, 10),
			Config:  inference.BackendConfiguration{},
		})
	}

	rr, _ := r.records.Get(runner)
	rr.Records = append(rr.Records, record)

	if len(rr.Records) > 10 {
		rr.Records = rr.Records[1:]
	}

	return recordID
}

func (r *OpenAIRecorder) NewResponseRecorder(w http.ResponseWriter) http.ResponseWriter {
	rc := &responseRecorder{
		ResponseWriter: w,
		body:           &bytes.Buffer{},
		statusCode:     http.StatusOK,
	}
	return rc
}

func (r *OpenAIRecorder) RecordResponse(id string, runner runnermap.Key, rw http.ResponseWriter) {
	rr := rw.(*responseRecorder)

	responseBody := rr.body.String()
	statusCode := rr.statusCode

	var response string
	if strings.Contains(responseBody, "data: ") {
		response = r.convertStreamingResponse(responseBody)
	} else {
		response = responseBody
	}

	r.m.Lock()
	defer r.m.Unlock()

	if modelData, exists := r.records.Get(runner); exists {
		for _, record := range modelData.Records {
			if record.ID == id {
				record.Response = response
				record.StatusCode = statusCode
				return
			}
		}
		r.log.Errorf("Matching request (id=%s) not found for model %s - %d\n%s", id, runner.Model, statusCode, response)
	} else {
		r.log.Errorf("Model %s not found in records - %d\n%s", runner.Model, statusCode, response)
	}
}

func (r *OpenAIRecorder) convertStreamingResponse(streamingBody string) string {
	lines := strings.Split(streamingBody, "\n")
	var contentBuilder strings.Builder
	var lastChunk map[string]interface{}

	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			lastChunk = chunk

			if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if delta, ok := choice["delta"].(map[string]interface{}); ok {
						if content, ok := delta["content"].(string); ok {
							contentBuilder.WriteString(content)
						}
					}
				}
			}
		}
	}

	if lastChunk == nil {
		return streamingBody
	}

	finalResponse := make(map[string]interface{})

	for key, value := range lastChunk {
		finalResponse[key] = value
	}

	if choices, ok := finalResponse["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			choice["message"] = map[string]interface{}{
				"role":    "assistant",
				"content": contentBuilder.String(),
			}
			delete(choice, "delta")

			if _, ok := choice["finish_reason"]; !ok {
				choice["finish_reason"] = "stop"
			}
		}
	}

	finalResponse["object"] = "chat.completion"

	jsonResult, err := json.Marshal(finalResponse)
	if err != nil {
		return streamingBody
	}

	return string(jsonResult)
}

func (r *OpenAIRecorder) GetRecordsByModelHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		model := req.URL.Query().Get("model")

		if model == "" {
			http.Error(w, "A 'model' query parameter is required", http.StatusBadRequest)
		} else {
			// Retrieve records for the specified model.
			modelData := r.GetModelData(model)
			if modelData == nil {
				// No records found for the specified model.
				http.Error(w, fmt.Sprintf("No records found for model '%s'", model), http.StatusNotFound)
				return
			}

			if err := json.NewEncoder(w).Encode(map[string]interface{}{
				"model":   model,
				"records": modelData.Records,
				"count":   len(modelData.Records),
				"config":  modelData.Config,
			}); err != nil {
				http.Error(w, fmt.Sprintf("Failed to encode records for model '%s': %v", model, err),
					http.StatusInternalServerError)
				return
			}
		}
	}
}

func (r *OpenAIRecorder) GetModelData(model string) *ModelData {
	r.m.RLock()
	defer r.m.RUnlock()

	if modelData, exists := r.records.Get(runnermap.Key{llamacpp.Name, model, inference.BackendModeCompletion}); exists {
		records := make([]*RequestResponsePair, len(modelData.Records))
		copy(records, modelData.Records)
		return &ModelData{Config: modelData.Config, Records: records}
	}

	return nil
}

func (r *OpenAIRecorder) RemoveModel(model string) {
	r.m.Lock()
	defer r.m.Unlock()

	runnerKey := runnermap.Key{llamacpp.Name, model, inference.BackendModeCompletion}
	if _, exists := r.records.Get(runnerKey); exists {
		r.records.Delete(runnerKey)
		r.log.Infof("Removed records for model: %s", model)
	} else {
		r.log.Warnf("No records found for model: %s", model)
	}
}
