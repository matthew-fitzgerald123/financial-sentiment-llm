output "api_url" {
  description = "Public URL of the load balancer"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository to push images to"
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "alarms_topic_arn" {
  description = "SNS topic ARN that CloudWatch alarms publish to"
  value       = aws_sns_topic.alarms.arn
}
