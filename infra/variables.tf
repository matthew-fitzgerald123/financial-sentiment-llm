variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Resource name prefix"
  type        = string
  default     = "financial-sentiment-llm"
}

variable "model_version" {
  description = "MODEL_VERSION env var injected into the container"
  type        = string
  default     = "mistral-7b-finance-mlx-lora-v1"
}

variable "task_cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)"
  type        = number
  default     = 2048
}

variable "task_memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 8192
}

variable "desired_count" {
  description = "Number of ECS tasks to run"
  type        = number
  default     = 1
}

variable "instance_type" {
  description = "EC2 instance type for the GPU ECS cluster (must have NVIDIA GPU for vLLM)"
  type        = string
  default     = "g4dn.xlarge"
}

variable "gpu_count" {
  description = "Number of GPUs to reserve per ECS task"
  type        = number
  default     = 1
}

variable "alarm_email" {
  description = "Email address to notify on CloudWatch alarms. Leave empty to skip the SNS email subscription."
  type        = string
  default     = ""
}
