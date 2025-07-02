package runnermap

import (
	"github.com/docker/model-runner/pkg/inference"
)

// Key is used to index runners.
type Key struct {
	// Backend is the Backend associated with the runner.
	Backend string
	// Model is the Model associated with the runner.
	Model string
	// Mode is the operation Mode associated with the runner.
	Mode inference.BackendMode
}

type Map[T any] struct {
	m            map[Key]T
	initialModel map[Key]string
	normalizeFn  func(string) string
}

func New[T any](normalizeFn func(string) string) *Map[T] {
	return &Map[T]{
		m:            make(map[Key]T),
		initialModel: make(map[Key]string),
		normalizeFn:  normalizeFn,
	}
}

func (rm *Map[T]) normalizeKey(key Key) Key {
	key.Model = rm.normalizeFn(key.Model)
	return key
}

func (rm *Map[T]) Set(key Key, value T) {
	normKey := rm.normalizeKey(key)
	rm.initialModel[normKey] = key.Model
	rm.m[normKey] = value
}

func (rm *Map[T]) Get(key Key) (T, bool) {
	normKey := rm.normalizeKey(key)
	val, ok := rm.m[normKey]
	return val, ok
}

func (rm *Map[T]) GetInitialModel(key Key) string {
	normKey := rm.normalizeKey(key)
	return rm.initialModel[normKey]
}

func (rm *Map[T]) Delete(key Key) {
	normKey := rm.normalizeKey(key)
	delete(rm.m, normKey)
	delete(rm.initialModel, normKey)
}

func (rm *Map[T]) Items() map[Key]T {
	rmCopy := make(map[Key]T, len(rm.m))
	for k, v := range rm.m {
		rmCopy[k] = v
	}
	return rmCopy
}
