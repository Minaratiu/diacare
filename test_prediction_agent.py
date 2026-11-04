
from src.prediction.prediction_agent import PredictionAgent


def main():
    agent = None

    try:
        print("DANG KHOI TAO PREDICTION AGENT...")

        agent = PredictionAgent()

        # Dữ liệu thử nghiệm gồm đúng 8 thuộc tính
        sample_data = {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 25,
            "Insulin": 80,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 30,
        }

        print("\nDU LIEU DAU VAO:")
        for name, value in sample_data.items():
            print(f"{name}: {value}")

        # Gọi Prediction Agent
        result = agent.predict(sample_data)

        print("\nKET QUA DU DOAN:")
        print("Success:", result["success"])
        print("Predicted class:", result["predicted_class"])
        print("Label:", result["label"])
        print("Model score:", result["model_score"])
        print("Threshold:", result["threshold"])
        print("Disclaimer:", result["disclaimer"])

        print("\nPREDICTION AGENT HOAT DONG!")

    except Exception as error:
        print("\nKIEM THU THAT BAI!")
        print("Chi tiet:", error)

    finally:
        # Prediction Agent không giữ kết nối cần đóng
        pass


if __name__ == "__main__":
    main()
