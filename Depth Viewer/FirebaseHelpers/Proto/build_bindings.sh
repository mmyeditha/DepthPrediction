#!/bin/bash

protoc --python_out=. Mesh.proto
protoc --swift_out=. Mesh.proto
