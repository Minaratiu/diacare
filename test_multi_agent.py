
from src.orchestration.coordinator import CoordinatorAgent


def main():
    coordinator = None

    try:
        print("DANG KHOI TAO DIACARE MULTI-AGENT...")

        coordinator = CoordinatorAgent()

        health_data = {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 25,
            "Insulin": 80,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 30,
        }

        question = "Nhung trieu chung thuong gap cua benh tieu duong la gi?"

        # TEST 1: Chỉ đánh giá dữ liệu sức khỏe
        print("\n===== TEST 1: PREDICTION =====")
        result1 = coordinator.process(health_data=health_data)
        print(result1.get("response", result1))

        # TEST 2: Chỉ tra cứu kiến thức
        print("\n===== TEST 2: KNOWLEDGE =====")
        result2 = coordinator.process(question=question)
        print(result2.get("response", result2))

        # TEST 3: Kết hợp cả hai chức năng
        print("\n===== TEST 3: COMBINED =====")
        result3 = coordinator.process(
            question=question,
            health_data=health_data
        )
        print(result3.get("response", result3))

        # Kiểm tra ba luồng
        if all([
            result1.get("success"),
            result2.get("success"),
            result3.get("success")
        ]):
            print("\nDIACARE MULTI-AGENT HOAT DONG!")
        else:
            print("\nCAN KIEM TRA LAI LUONG XU LY!")

    except Exception as error:
        print("\nKIEM THU THAT BAI!")
        print("Chi tiet:", error)

    finally:
        if coordinator is not None:
            coordinator.close()


if __name__ == "__main__":
    main()
