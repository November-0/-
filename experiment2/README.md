# 实验2_创建首个Kotlin应用

## 1 创建一个新的工程
打开Android Studio，选择Projects>New Project，然后选择Basic Activity.
![1](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/1.png)\
点击Next，为应用程序命名（例如：MyApp2），选择Kotlin语言，然后点击Finish。Android Studio将使用系统中最新的API Level创建应用程序，并使用Gradle作为构建系统，在底部的视窗中可以查看整个过程。
![2](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/2.png)\
## 2 在模拟器上运行应用程序
选择Run>Run ‘app’，在工具栏上可以看到运行程序的一些选择项。
![3](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/3.png)\
运行效果：
![4](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/4.png)\
## 3 查看布局编辑器
在Basic Activity中，包含了基本的导航组件，Android app关联两个fragments，第一个屏幕显示了“Hello first fragment”由FirstFragment创建，界面元素的排列由布局文件指定，查看res>layout>fragment_first.xml，
![5](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/5.png)\
查看布局的代码（Code），修改Textview的Text属性，
```
android:text="@string/hello_first_fragment"
```
右键该代码，选择Go To > Declaration or Usages，跳转到values/strings.xml，看到高亮文本
```
<string name="hello_first_fragment">Hello first fragment</string>
```
修改字符串属性值为“`Hello Kotlin!`”。更进一步，修改字体显示属性，在Design视图中选择textview_first文本组件，在Common Attributes属性下的textAppearance域，设置相关的文字显示属性，
![6](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/6.png)\
查看布局的XML代码，可以看到新属性被应用。
```
android:fontFamily="sans-serif-condensed"
android:text="@string/hello_first_fragment"
android:textColor="@android:color/darker_gray"
android:textSize="30sp"
android:textStyle="bold"
```
## 4 向页面添加更多的布局
本步骤将向第一个Fragment添加更多的视图组件。
### 4.1 查看视图的布局约束
在fragment_first.xml，查看TextView组件的约束属性：
![7](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/7.png)\
### 4.2 添加按钮和约束
从Palette面板中拖动Button到
![8](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/8.png)\
调整Button的约束，设置Button的Top>BottonOf textView，
```
app:layout_constraintTop_toBottomOf="@+id/textview_first" />
```
随后添加Button的左侧约束至屏幕的左侧，Button的底部约束至屏幕的底部。查看Attributes面板，修改将id从button修改为toast_button（==注意修改id将重构代码==）。
### 4.3 调整Next按钮
Next按钮是工程创建时默认的按钮，查看Next按钮的布局设计视图，它与TextView之间的连接不是锯齿状的而是波浪状的，表明两者之间存在链（chain），是一种两个组件之间的双向联系而不是单向联系。删除两者之间的链，可以在设计视图右键相应约束，选择Delete（注意两个组件要双向删除）。
![9](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/9.png)\
同时，删除Next按钮的左侧约束。
### 4.4 添加新的约束
添加Next的右边和底部约束至父类屏幕（如果不存在的话），Next的Top约束至TextView的底部。最后，TextView的底部约束至屏幕的底部。效果看起来如下图所示：
![10](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/10.png)\
### 4.5 更改组件的文本
fragment_first.xml布局文件代码中，找到toast_button按钮的text属性部分
```
<Button
   android:id="@+id/toast_button"
   android:layout_width="wrap_content"
   android:layout_height="wrap_content"
   android:text="Button"
```
这里text的赋值是一种硬编码，点击文本，左侧出现灯泡状的提示，选择 Extract string resource。
![11](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/11.png))
弹出对话框，令资源名为toast_button_text，资源值为Toast，并点击OK。
![12](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/12.png))
于是，在资源文件string.xml定义了字符串，以上操作可以手动在string.xml文件中定义并引用。
```
<resources>
   ... 
   <string name="toast_button_text">Toast</string>
</resources>
```
### 4.6 更新Next按钮
在属性面板中更改Next按钮的id，从button_first改为random_button。
![13](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/13.png)\
在string.xml文件，右键next字符串资源，选择 Refactor > Rename，修改资源名称为random_button_text，点击Refactor 。随后，修改Next值为Random。
### 4.7 添加第三个按钮
向fragment_first.xml文件中添加第三个按钮，位于Toast和Random按钮之间，TextView的下方。新Button的左右约束分别约束至Toast和Random，Top约束至TextView的底部，Buttom约束至屏幕的底部，看起来的效果：
![14](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/14.png)\
检查xml代码，确保不出现类似app:layout_constraintVertical_bias这样的属性，即不手动设置偏移量。

### 4.8 完善UI组件的属性设置
更改新增按钮id为count_button，显示字符串为Count，对应字符串资源值为count_button_text。
同时，更改TextView的文本为0。修改后的fragment_first.xml的代码如下：
```
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:background="@color/screenBackground"
    tools:context=".FirstFragment">

    <TextView
        android:id="@+id/textview_first"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:fontFamily="sans-serif-condensed"
        android:text="@string/hello_first_fragment"
        android:textColor="@android:color/white"
        android:textSize="72sp"
        android:textStyle="bold"
        app:layout_constraintVertical_bias="0.3"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent" />

    <Button
        android:id="@+id/random_button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginEnd="24dp"
        android:background="@color/buttonBackground"
        android:text="@string/random_button_text"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintTop_toBottomOf="@+id/textview_first" />

    <Button
        android:id="@+id/toast_button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginStart="24dp"
        android:background="@color/buttonBackground"
        android:text="@string/toast_button_text"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@+id/textview_first" />

    <Button
        android:id="@+id/count_button"
        android:background="@color/buttonBackground"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="@string/count_button_text"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toStartOf="@+id/random_button"
        app:layout_constraintStart_toEndOf="@+id/toast_button"
        app:layout_constraintTop_toBottomOf="@+id/textview_first" />
</androidx.constraintlayout.widget.ConstraintLayout>
```
## 5 更新按钮和文本框的外观
### 5.1 添加新的颜色资源
values>colors.xml定义了一些应用程序可以使用的颜色，添加新颜色screenBackground 值为 #2196F3，这是蓝色阴影色；添加新颜色buttonBackground 值为 #BBDEFB
```
<color name="screenBackground">#2196F3</color>
<color name="buttonBackground">#BBDEFB</color>
```
### 5.2 设置组件的外观
①fragment_first.xml的属性面板中设置屏幕背景色为
```
android:background="@color/screenBackground"
```
②设置每个按钮的背景色为buttonBackground
```
android:background="@color/buttonBackground"
```
注意：需修改res/values/themes.xml的style值，添加**.Bridge**。
```
<style name="Theme.MyFirstApp" parent="Theme.MaterialComponents.DayNight.DarkActionBar.Bridge">
```
③移除TextView的背景颜色，设置TextView的文本颜色为color/white，并增大字体大小至72sp
### 5.3 设置组件的位置
①Toast与屏幕的左边距设置为24dp，Random与屏幕的右边距设置为24dp，利用属性面板的Constraint Widget完成设置。
![15](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/15.png)\
\
![16](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/16.png)\
②设置TextView的垂直偏移为0.3，
```
app:layout_constraintVertical_bias="0.3"
```
拖动左侧的移动条。
![17](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/17.png)\
### 5.4 运行应用程序
最终效果如下图：
![18](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/18.png)\


## 6 添加代码完成应用程序交互
### 6.1 设置代码自动补全
Android Studio中，依次点击File>New Projects Settings>Settings for New Projects…，查找Auto Import选项，在Java和Kotlin部分，勾选**Add Unambiguous Imports on the fly**。
![19](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/19.png)\
### 6.2 TOAST按钮添加一个toast消息
打开FirstFragment.kt文件，有三个方法：onCreateView，onViewCreated和onDestroyView，在onViewCreated方法中使用绑定机制设置按钮的响应事件（创建应用程序时自带的按钮）。
```
binding.randomButton.setOnClickListener {
    findNavController().navigate(R.id.action_FirstFragment_to_SecondFragment)
}
```
接下来，为TOAST按钮添加事件，使用**findViewById()**查找按钮id，代码如下：
```
// find the toast_button by its ID and set a click listener
view.findViewById<Button>(R.id.toast_button).setOnClickListener {
   // create a Toast with some text, to appear for a short time
   val myToast = Toast.makeText(context, "Hello Toast!", Toast.LENGTH_LONG)
   // show the Toast
   myToast.show()
}
```
代码使用了Lambda表达式的机制。
### 6.3 使Count按钮更新屏幕的数字
此步骤向Count按钮添加事件响应，更新Textview的文本显示。
在FirstFragment.kt文件，为count_buttion按钮添加事件：
```
view.findViewById<Button>(R.id.count_button).setOnClickListener {
   countMe(view)
}
```
countMe()为自定义方法，以View为参数，每次点击增加数字1，具体代码为：
```
private fun countMe(view: View) {
   // Get the text view
   val showCountTextView = view.findViewById<TextView>(R.id.textview_first)

   // Get the value of the text view.
   val countString = showCountTextView.text.toString()

   // Convert value to a number and increment it
   var count = countString.toInt()
   count++

   // Display the new value in the text view.
   showCountTextView.text = count.toString()
}
```
### 6.4 运行应用程序
最终效果如下图：
点击TOAST按钮
![20](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/20.png)\
点击Count按钮
![21](https://raw.githubusercontent.com/November-0/Software-project-R-amp-D-practice/main/experiment2/images/21.png)\
