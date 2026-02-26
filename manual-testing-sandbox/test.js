class Calculator {
  constructor() {
    this.result = 0;
  }

  add(number) {
    this.result += number;
    return this;
  }
  subtract(number) {
    this.result -= number;
    return this;
  }

  multiply(number) {
    this.result *= number;
    return this;
  }

  divide(number) {
    if (number === 0) {
      throw new Error("Cannot divide by zero");
    }
    this.result /= number;
    return this;
  }

  getResult() {
    return this.result;
  }

  reset() {
    this.result = 0;
    return this;
  }
}

// 测试代码
const calc = new Calculator();
console.log(calc.add(5).multiply(2).getResult()); // 应该输出 10

// 更多测试
const calc2 = new Calculator();
calc2.add(10).subtract(3).multiply(4).divide(2);
console.log(calc2.getResult()); // 应该输出 14

// 测试除法错误
try {
  const calc3 = new Calculator();
  calc3.add(10).divide(0);
} catch (error) {
  console.log("捕获到错误:", error.message);
}
