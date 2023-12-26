#![feature(absolute_path)]

use std::io::Write;

use anyhow::{anyhow, Result};
use async_openai::Client;
use async_openai::types::{ChatCompletionFunctionsArgs, ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, ChatCompletionToolArgs, CreateChatCompletionRequestArgs};
use base64::Engine;
use base64::engine::general_purpose;
use serde::Deserialize;
use serde_json::json;
use tokio::io::AsyncBufReadExt;

async fn generate_picture(
    fooocus_api: &str,
    http_client: &reqwest::Client,
    prompts: &str,
) -> Result<String> {
    #[derive(Deserialize)]
    struct Prompts {
        prompts: String,
    }

    #[derive(Deserialize)]
    struct Message {
        base64: String,
    }

    let prompts: Prompts = serde_json::from_str(prompts)?;
    let prompts = prompts.prompts.as_str();

    let json: Vec<Message> = http_client.post(fooocus_api)
        .json(&json!({
            "base_model_name": "11-27_CHEYENNE_v10.safetensors",
            "prompt": prompts,
            "negative_prompt": "(embedding:unaestheticXLv31:0.8), low quality, watermark,verybadimagenegative_v1.3,ng_deepnegative_v1_75t,EasyNegative,badhandv4,rev2-badprompt,easynegative",
            "style_selections": [
            ],
            "advanced_params": {
            },
            "require_base64": true,
            "async_process": false
        }))
        .send()
        .await?
        .json()
        .await?;

    let img_bytes = general_purpose::STANDARD.decode(&json.get(0).ok_or_else(|| anyhow!("message is empty"))?.base64)?;
    let random: u32 = rand::random();
    let path = std::path::absolute(format!("{}-{}.png", prompts, random))?;

    tokio::fs::write(&path, img_bytes).await?;
    Ok(path.to_str().unwrap().to_string())
}

async fn exec(fooocus_api: &str) -> Result<()> {
    let client = Client::new();
    let http_client = reqwest::Client::new();
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());

    let func_args = ChatCompletionFunctionsArgs::default()
        .name("generate_picture")
        .description("Generate picture from prompt words")
        .parameters(json!({
                "type": "object",
                "properties": {
                    "prompts": {
                        "type": "string",
                        "description": "Prompt words used to generate picture, such as \"A cat\"",
                    }
                },
                "required": ["prompts"],
            }))
        .build()?;

    let tools = vec![
        ChatCompletionToolArgs::default()
            .function(func_args)
            .build()?
    ];

    let mut messages = Vec::new();

    let message = ChatCompletionRequestSystemMessageArgs::default()
        .content("Your name is Fooocus, Help me generate pictures")
        .build()?;

    messages.push(ChatCompletionRequestMessage::from(message));

    let mut skip = false;
    let mut line = String::new();

    loop {
        if !skip {
            print!("User: ");
            std::io::stdout().flush()?;

            line.clear();
            stdin.read_line(&mut line).await?;

            let message = ChatCompletionRequestUserMessageArgs::default()
                .content(line.as_str())
                .build()?;

            messages.push(ChatCompletionRequestMessage::from(message));
        }

        skip = false;

        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(512u16)
            .model("gpt-3.5-turbo")
            .messages(messages.as_slice())
            .tools(tools.as_slice())
            .build()?;

        let resp = client.chat().create(request).await?;

        for choice in resp.choices {
            let resp_message = choice.message;

            let mut message_args = ChatCompletionRequestAssistantMessageArgs::default();

            if let Some(content) = &resp_message.content {
                message_args.content(content);
            };

            if let Some(tool_calls) = &resp_message.tool_calls {
                message_args.tool_calls(tool_calls.clone());
            }

            messages.push(ChatCompletionRequestMessage::from(message_args.build()?));

            if let Some(content) = resp_message.content {
                println!("Fooocus: {}", content);
            }

            if let Some(tool) = resp_message.tool_calls {
                for call in tool {
                    let func = call.function;

                    assert_eq!(func.name, "generate_picture");
                    let res = match generate_picture(fooocus_api, &http_client, &func.arguments).await {
                        Ok(path) => {
                            opener::open(&path)?;

                            json!({
                                "picture_path": path
                            })
                        }
                        Err(e) => {
                            json!({
                                "error": e.to_string()
                            })
                        }
                    };

                    let message = ChatCompletionRequestToolMessageArgs::default()
                        .content(res.to_string())
                        .tool_call_id(call.id)
                        .build()?;

                    messages.push(ChatCompletionRequestMessage::from(message));
                    skip = true;
                }
            }
        }
    }
}

fn main() {
    // https://xxxx/v1/generation/text-to-image
    let fooocus_api = std::env::var("FOOOCUS_API").expect("must need FOOOCUS_API env");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(exec(&fooocus_api)).unwrap();
}