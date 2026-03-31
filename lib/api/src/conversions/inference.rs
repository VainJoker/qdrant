use tonic::Status;

use crate::conversions::json::{dict_to_proto, json_to_proto, proto_dict_to_json, proto_to_json};
use crate::grpc::qdrant as grpc;
use crate::rest::{DocumentOptions, Options, schema as rest};

impl From<rest::Document> for grpc::Document {
    fn from(document: rest::Document) -> Self {
        let rest::Document {
            text,
            model,
            options,
            fuzzy,
        } = document;
        Self {
            text,
            model,
            fuzzy: fuzzy.map(|f| grpc::FuzzyParams {
                max_edits: Some(f.max_edits as u32),
                prefix_length: Some(f.prefix_length as u32),
                max_expansions: Some(f.max_expansions as u32),
            }),
            options: options
                .map(DocumentOptions::into_options)
                .map(dict_to_proto)
                .unwrap_or_default(),
        }
    }
}

impl TryFrom<grpc::Document> for rest::Document {
    type Error = Status;

    fn try_from(document: grpc::Document) -> Result<Self, Self::Error> {
        let grpc::Document {
            text,
            model,
            options,
            fuzzy,
        } = document;
        Ok(Self {
            text,
            model,
            fuzzy: fuzzy.map(|f| segment::types::FuzzyParams {
                max_edits: f.max_edits.unwrap_or(0) as u8,
                prefix_length: f.prefix_length.unwrap_or(0) as u8,
                max_expansions: f.max_expansions.unwrap_or(0) as u8,
            }),
            options: Some(DocumentOptions::Common(proto_dict_to_json(options)?)),
        })
    }
}

impl From<rest::Image> for grpc::Image {
    fn from(image: rest::Image) -> Self {
        let rest::Image {
            image,
            model,
            options,
        } = image;
        Self {
            image: Some(json_to_proto(image)),
            model,
            options: options.options.map(dict_to_proto).unwrap_or_default(),
        }
    }
}

impl TryFrom<grpc::Image> for rest::Image {
    type Error = Status;

    fn try_from(image: grpc::Image) -> Result<Self, Self::Error> {
        let grpc::Image {
            image,
            model,
            options,
        } = image;

        let image = image.ok_or_else(|| Status::invalid_argument("Empty image is not allowed"))?;

        Ok(Self {
            image: proto_to_json(image)?,
            model,
            options: Options {
                options: Some(proto_dict_to_json(options)?),
            },
        })
    }
}

impl From<rest::InferenceObject> for grpc::InferenceObject {
    fn from(object: rest::InferenceObject) -> Self {
        let rest::InferenceObject {
            object,
            model,
            options,
        } = object;
        Self {
            object: Some(json_to_proto(object)),
            model,
            options: options.options.map(dict_to_proto).unwrap_or_default(),
        }
    }
}

impl TryFrom<grpc::InferenceObject> for rest::InferenceObject {
    type Error = Status;

    fn try_from(object: grpc::InferenceObject) -> Result<Self, Self::Error> {
        let grpc::InferenceObject {
            object,
            model,
            options,
        } = object;

        let object =
            object.ok_or_else(|| Status::invalid_argument("Empty object is not allowed"))?;

        Ok(Self {
            object: proto_to_json(object)?,
            model,
            options: Options {
                options: Some(proto_dict_to_json(options)?),
            },
        })
    }
}
