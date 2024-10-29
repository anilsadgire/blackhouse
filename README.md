# blackhouse
# Welcome to the Machine Learning Engineer Work Trial

Hello and welcome! This work trial is designed to assess your skills in implementing a research paper from start to finish and deploying the model on AWS for generating trade recommendations. Your performance in this task will give us insights into your ability to tackle complex machine learning challenges and contribute effectively to our team.

## Paper to Implement
We're diving into the fascinating world of reinforcement learning with the paper: [Reinforcement Learning in Limit Order Markets](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf). This foundational research will guide your implementation.

## Codebase
To kick things off, here’s the [GitHub Repository](https://github.com/Blockhouse-Repo/Blockhouse-Work-Trial) you’ll be working with. Inside, you’ll find:
- The dataset you’ll need for your model.
- Code snippets for different costs in the rewards function.
- Baseline strategies using TWAP and VWAP.

## Task Objectives
Your mission, should you choose to accept it, involves several key objectives:

1. **Build a SELL-side Model**: Address this problem statement: “I want to sell 1000 shares of AAPL in one day. How should I split the trades to optimize transaction costs?” Remember, you have 390 minutes (one trading day) to work with!

2. **Focus on Minimizing Transaction Costs**: Unlike the paper's goal of maximizing wealth returns, your focus will be on minimizing transaction costs specifically for SELL-side implementation.

3. **Use Minute-Wise Data**: The dataset has been processed to be in minute-wise format, which you’ll utilize instead of the millisecond data mentioned in the paper.

4. **Deliver a Trade Schedule**: Your final output should include:
   - **Timestamp**: When to execute the trade.
   - **Share Size**: How many shares to sell at that timestamp.
   - Assume all trades are executed as Market Orders.

5. **Backtest Your Model**: Validate it on a testing set and compare its performance against TWAP and VWAP benchmarks.

6. **Deploy on AWS**: Create a SageMaker endpoint for real-time inference.

7. **Create a Demo Video**: Showcase your implementation, including a live demonstration of calling the endpoint and receiving the results in JSON format.

### Bonus Points
- If your model outperforms TWAP and VWAP on backtests, that’s a win!
- Implement logic for submitting limit orders along with the price point.

## Helpful Resources
Before you get started, check out these resources to help you implement the reinforcement learning model, optimize for transaction costs, and deploy on AWS:

1. **Reinforcement Learning in Limit Order Markets**: The primary paper you’ll be replicating, adjusted for minimizing transaction costs.

2. **AWS SageMaker Documentation**: 
   - [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html): Learn how to deploy your model on SageMaker, including setting up real-time inference endpoints.
   - [SageMaker Real-Time Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/real-time-endpoints.html): Detailed guide for deploying ML models with real-time inference.

3. **Soft Actor-Critic (SAC) Algorithm**: 
   - [Soft Actor-Critic Paper](https://arxiv.org/abs/1812.05905): Understand the SAC algorithm.
   - [Stable Baselines3 SAC Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html): A handy library for implementing SAC in Python with PyTorch.

4. **Understanding TWAP and VWAP**: 
   - [TWAP vs VWAP](https://chain.link/education-hub/twap-vs-vwap) will clarify these important strategies.

5. **Python Libraries for RL and Trading**: 
   - [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html): A scalable library for reinforcement learning.
   - [Pandas for Financial Data](https://pandas.pydata.org/pandas-docs/stable/index.html): Essential for managing financial datasets and performing time-series operations.

## Breakdown of the Task

### Setup and Environment
- Use Python along with relevant libraries like PyTorch, TensorFlow, or RLlib for your reinforcement learning needs.
- Leverage AWS SageMaker for deploying the model and enabling real-time inference.

### Model Implementation
1. **Model Replication**: 
   - Implement the reinforcement learning model as per the research paper.
   - Adjust the model to focus on minimizing transaction costs.

2. **Model Fine-Tuning**: 
   - Fine-tune your model using the provided market data.
   - Experiment with various hyperparameters like learning rate, reward structure, and discount factor to optimize performance.

### Back-Testing and Benchmarking
- Backtest your model using the trading data.
- Compare its performance against TWAP and VWAP strategies, focusing on slippage and market impact.

### AWS Deployment
1. **Create a Real-Time Endpoint**: 
   - Deploy your fine-tuned model on AWS SageMaker as a real-time inference endpoint.
   - Ensure it can accept live trade data and provide optimized sell schedules (timestamp + share size).

2. **API Endpoint**: 
   - Create an API endpoint that takes the following parameters:
     - Ticker (AAPL).
     - Number of shares to sell.
     - Default time horizon for execution (390 minutes, or one trading day).
   - The endpoint should return an optimized schedule in JSON format.

### Documentation and Video Submission
- Provide a detailed written report covering:
  - Model architecture.
  - Fine-tuning strategies.
  - Back-testing results.
  - The AWS deployment process.
- Submit a video demonstrating:
  - A live demo of calling the SageMaker endpoint and receiving the trade schedule.

## Evaluation Criteria
We’ll be looking at several factors to evaluate your work:

1. **Innovation and Model Implementation**: 
   - Use of appropriate techniques (like DQN or PPO) to solve the problem.
   - Adherence to the goal of minimizing transaction costs.

2. **Code Quality and Documentation**: 
   - Clarity, structure, and readability of your code.
   - Detailed documentation explaining your implementation and deployment steps.

3. **Real-Time Deployment and Usability**: 
   - Successful deployment via AWS SageMaker.
   - Demonstration of API usage with inputs and outputs in JSON format.

4. **Performance Evaluation**: 
   - Back-testing results and comparison against TWAP and VWAP.
   - Effective deployment for real-time inference.

## Conclusion
This work trial is designed to mirror the challenges and tasks you would face as a Machine Learning Engineer at Blockhouse. It offers a great opportunity to showcase your technical expertise, innovative thinking, and problem-solving skills in a real-world scenario. We’re excited to see your innovative solutions and hope to potentially welcome you to our team!

Good luck, and let’s get started!
